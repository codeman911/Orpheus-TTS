from orpheus_tts import OrpheusModel
import struct
import io
import wave
import time
import re

from fastapi import FastAPI, HTTPException, UploadFile, Form, File, WebSocket, WebSocketDisconnect
from typing import Literal, Optional
from fastapi.middleware.cors import CORSMiddleware


from fastapi.responses import StreamingResponse
import uvicorn
import logging
logging.basicConfig(level=logging.INFO)
_LOG = logging.getLogger(__name__)

model = OrpheusModel("/vast/audio/experiment/Orpheus-TTS/speaker_finetune/checkpoints_spk/checkpoint-15000")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


def create_wav_header(sample_rate=24000, bits_per_sample=16, channels=1):
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8

    data_size = 0

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header


@app.post("/generate_audio", response_class=StreamingResponse)
async def generate_endpoint(text: str = Form(...), voice: Optional[str] = Form(None)):
    #prompt = "قبل كنا نقعد مع الأهل والسوالف ما تخلص، كل حد يضحك ويشارك قصصه، أما الحين، كل واحد ماسك تلفونه وما حد يدري عن الثاني."
        _LOG.info(
            f"Generating audio from processed text ({len(text)} chars, voice {voice}): {text}"
        )

        def generate_audio_stream(text, voice, request_id="req-001", repetition_penalty=1.3, max_tokens=100000,
                                  temperature=0.7, top_p=0.9):
            yield create_wav_header()

            for sentence in re.split(r'(?<=[.!؟?])\s+', text):
                audio_generator = model.generate_speech(
                    prompt=sentence,
                    voice=voice,
                    request_id=request_id,
                    repetition_penalty=repetition_penalty,
                    stop_token_ids=[128258],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                for chunk in audio_generator:
                    yield chunk

                duration_sec = 0.15
                sample_rate = 24000
                num_samples = int(sample_rate * duration_sec)
                silence = struct.pack("<h", 0) * num_samples
                yield silence

        return StreamingResponse(generate_audio_stream(text, voice), media_type="audio/wav")


@app.post("/generate_audio_zs", response_class=StreamingResponse)
async def generate_endpoint_zero_shot(text: str = Form(...),
                                      reference_text: str = Form(...),
                                      reference_audio: UploadFile = File(...)):
        _LOG.info(
            f"Generating audio from processed text ({len(text)} chars, with reference audio: {text}"
        )

        audio_bytes = await reference_audio.read()

        def generate_audio_stream(text, voice, request_id="req-001", repetition_penalty=1.1, max_tokens=10000,
                                  temperature=0.4, top_p=0.9, reference_audio=None, reference_text=None):
            yield create_wav_header()

            for sentence in re.split(r'(?<=[.!؟?])\s+', text):
                audio_generator = model.generate_speech(
                    prompt=sentence,
                    voice=None,
                    reference_audio=reference_audio,
                    reference_text=reference_text,
                    request_id=request_id,
                    repetition_penalty=repetition_penalty,
                    stop_token_ids=[128258],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                for chunk in audio_generator:
                    yield chunk

                duration_sec = 0.15
                sample_rate = 24000
                num_samples = int(sample_rate * duration_sec)
                silence = struct.pack("<h", 0) * num_samples
                yield silence

        return StreamingResponse(generate_audio_stream(text, voice=None,
                                                       reference_audio=audio_bytes,
                                                       reference_text=reference_text), media_type="audio/wav")


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
