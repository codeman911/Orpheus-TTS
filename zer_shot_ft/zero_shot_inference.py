import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import argparse
import wave
import time
import os
from orpheus_tts import OrpheusModel, tokens_decoder_sync
from snac import SNAC
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ZeroShotVoiceCloner:
    def __init__(self, model_path, device=None):
        """Initialize the voice cloner with model and tokenizer"""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load Orpheus TTS model
        logger.info(f"Loading Orpheus TTS model from: {model_path}")
        self.model = OrpheusModel(model_name=model_path)
        
        # Load SNAC model for audio tokenization
        logger.info("Loading SNAC model for audio encoding")
        self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
        if torch.cuda.is_available():
            try:
                self.snac_model = self.snac_model.to("cuda")
            except Exception as e:
                logger.warning(f"Could not move SNAC model to CUDA, keeping on CPU: {e}")
        
        # Define special tokens - match exactly with training
        self.special_tokens = {
            "start_of_human": 128259,
            "end_of_human": 128260,
            "start_of_ai": 128261,
            "start_of_speech": 128257,
            "end_of_speech": 128258,
            "end_of_ai": 128262,
            "end_of_text": 128009,
            "audio_tokens_start": 128266
        }
    
    def load_reference_audio(self, audio_path, target_sr=24000):
        """Load and preprocess reference audio file"""
        logger.info(f"Loading reference audio from: {audio_path}")
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sample_rate != target_sr:
            resample_transform = T.Resample(orig_freq=sample_rate, new_freq=target_sr)
            waveform = resample_transform(waveform)
        
        return waveform, target_sr
    
    def tokenize_audio(self, waveform):
        """Convert audio waveform to SNAC tokens"""
        logger.info("Tokenizing audio waveform")
        
        # Prepare waveform for SNAC model
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform).unsqueeze(0).unsqueeze(0)
        elif isinstance(waveform, torch.Tensor) and waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)
        
        waveform = waveform.to(dtype=torch.float32)
        
        # Only move to GPU if SNAC model is on GPU
        if next(self.snac_model.parameters()).device.type == "cuda":
            waveform = waveform.to("cuda")
        
        # Encode audio using SNAC
        with torch.inference_mode():
            codes = self.snac_model.encode(waveform)
        
        # Process audio codes - match exactly with training format
        audio_tokens_start = self.special_tokens["audio_tokens_start"]
        reference_codes = []
        
        # Process in batches to reduce memory usage
        for i in range(codes[0].shape[1]):
            reference_codes.append(codes[0][0][i].item() + audio_tokens_start)
            reference_codes.append(codes[1][0][2*i].item() + audio_tokens_start + 4096)
            reference_codes.append(codes[2][0][4*i].item() + audio_tokens_start + (2*4096))
            reference_codes.append(codes[2][0][(4*i)+1].item() + audio_tokens_start + (3*4096))
            reference_codes.append(codes[1][0][(2*i)+1].item() + audio_tokens_start + (4*4096))
            reference_codes.append(codes[2][0][(4*i)+2].item() + audio_tokens_start + (5*4096))
            reference_codes.append(codes[2][0][(4*i)+3].item() + audio_tokens_start + (6*4096))
        
        logger.info(f"Generated {len(reference_codes)} audio tokens")
        return reference_codes
    
    def create_zero_shot_prompt(self, reference_audio, reference_text, target_text):
        """Create a prompt with reference audio for zero-shot voice cloning"""
        logger.info("Creating zero-shot prompt")
        
        # Load and process reference audio
        waveform, _ = self.load_reference_audio(reference_audio)
        
        # Tokenize audio
        reference_codes = self.tokenize_audio(waveform)
        
        # Get tokenizer from model
        tokenizer = self.model.tokeniser
        
        # Tokenize texts with return_tensors="pt"
        reference_text_ids = tokenizer(reference_text, return_tensors="pt").input_ids[0]
        target_text_ids = tokenizer(target_text, return_tensors="pt").input_ids[0]
        
        # Create prompt with correct dimensions
        start_tokens = torch.tensor([self.special_tokens["start_of_human"]], dtype=torch.int64)
        end_tokens = torch.tensor([
            self.special_tokens["end_of_text"],
            self.special_tokens["end_of_human"],
            self.special_tokens["start_of_ai"],
            self.special_tokens["start_of_speech"]
        ], dtype=torch.int64)
        final_tokens = torch.tensor([
            self.special_tokens["end_of_speech"],
            self.special_tokens["end_of_ai"]
        ], dtype=torch.int64)
        
        # Convert all to 1D tensors
        reference_codes_tensor = torch.tensor(reference_codes, dtype=torch.int64)
        end_tokens_final = torch.tensor([self.special_tokens["end_of_text"], self.special_tokens["end_of_human"]], dtype=torch.int64)
        
        # Concatenate with matching dimensions
        zeroprompt_input_ids = torch.cat([
            start_tokens,
            reference_text_ids,
            end_tokens,
            reference_codes_tensor,
            final_tokens,
            start_tokens,
            target_text_ids,
            end_tokens_final
        ], dim=0)
        
        return zeroprompt_input_ids
    
    def generate_speech(self, reference_audio, reference_text, target_text, output_path="output.wav", 
                        temperature=0.7, repetition_penalty=1.2):
        # Create the zero-shot prompt
        input_ids = self.create_zero_shot_prompt(reference_audio, reference_text, target_text)
        
        logger.info(f"Generating speech for: '{target_text}'")
        start_time = time.monotonic()
        
        # Convert to string for generation, removing <s> token if present
        prompt_text = self.model.tokeniser.decode(input_ids)
        prompt_text = prompt_text.replace("<s>", "")
        
        # Generate using model's generate_tokens_sync method
        token_generator = self.model.generate_tokens_sync(
            prompt=prompt_text,
            temperature=temperature,
            top_p=0.95,
            repetition_penalty=repetition_penalty,
            stop_token_ids=[self.special_tokens["end_of_speech"], self.special_tokens["end_of_ai"]],
            max_tokens=4096  # Fixed value instead of using config
        )
        
        # Process audio chunks
        audio_chunks = tokens_decoder_sync(token_generator)
        
        # Save audio
        logger.info(f"Saving audio to {output_path}")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            
            total_frames = 0
            for audio_chunk in audio_chunks:
                if audio_chunk:  # Check if chunk is not empty
                    frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
                    total_frames += frame_count
                    wf.writeframes(audio_chunk)
            
            duration = total_frames / wf.getframerate() if total_frames > 0 else 0
        
        end_time = time.monotonic()
        logger.info(f"Generated {duration:.2f}s audio")
        if duration > 0:
            logger.info(f"Generation took {end_time - start_time:.2f}s (RTF: {(end_time - start_time)/duration:.2f}x)")
        else:
            logger.info(f"Generation took {end_time - start_time:.2f}s (No audio generated)")
        
        return output_path

def main():
    parser = argparse.ArgumentParser(description='Zero-Shot Voice Cloning with Orpheus TTS')
    parser.add_argument('--model', type=str, 
                        default="/vast/audio/experiment/Orpheus-TTS/zer_shot_ft/checkpoints_zr/checkpoint-12000", 
                        help='Path to the Orpheus model checkpoint')
    parser.add_argument('--reference_audio', type=str, required=True, 
                        help='Path to reference audio file')
    parser.add_argument('--reference_text', type=str, 
                        default="""اه يعني رياضة أو شيء. فأختي خبرتني قالت لي اسمع في شيء اسمه بايليت رياضة بتعيبك. فيوم صرت المدربة يعني كانت ممتازة ويمكن تعرف لأني أنا كنت محتاج هذاك الوقت لأن معروف إن الرياضة طلع هرمونات السعادة والمدربة كانت وايد وايد ممتازة. يعني أنا انبهرت.""",
                        help='Transcript of the reference audio')
    parser.add_argument('--target_text', type=str, 
                        default="""ويمكن موضوع الملوحة في الماء. بس أنت يعني تعطينا Tips في صناعة المحتوى الزراعي عن كيف أننا نستعمل هذه التقنيات وكيف نستعمل، كيف نتعامل مع الفشل في الزراعة. شو ممكن تخبرنا عن صناعة المحتوى؟""" ,
                        # default="I finally got into the university of my dreams! I can't believe all this hard work actually paid off!", 
                        help='Text to synthesize in the cloned voice')
    parser.add_argument('--output', type=str, 
                        default="../cloned_voice.wav", 
                        help='Path to save the generated audio')
    parser.add_argument('--temperature', type=float, default=0.9,
                        help='Sampling temperature')
    parser.add_argument('--repetition_penalty', type=float, default=1.3,
                        help='Repetition penalty')
    args = parser.parse_args()
    
    # Check if reference audio exists
    if not os.path.exists(args.reference_audio):
        logger.error(f"Reference audio file not found: {args.reference_audio}")
        return
    
    # Initialize the voice cloner
    cloner = ZeroShotVoiceCloner(model_path=args.model)
    
    # Generate speech with the cloned voice
    cloner.generate_speech(
        reference_audio=args.reference_audio,
        reference_text=args.reference_text,
        target_text=args.target_text,
        output_path=args.output,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty
    )
    
    logger.info(f"Audio saved to: {args.output}")

if __name__ == "__main__":
    main()