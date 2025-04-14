import json
import os
import re
import torch
import torchaudio
import torchaudio.functional as taf
from pathlib import Path
from tqdm import tqdm
import argparse
import logging

logging.basicConfig(level=logging.INFO)
_LOG = logging.getLogger(__name__)


def segment_audio_by_time(waveform, sample_rate, start_time_second, end_time_second, tgt_af, tgt_sample_rate=24000,
                          bits_per_sample=32):
    start_sample = int(start_time_second * sample_rate)
    end_sample = int(end_time_second * sample_rate)

    segmented_audio = waveform[:, start_sample:end_sample]

    if segmented_audio.shape[0] > 1:
        segmented_audio = torch.mean(segmented_audio, dim=0, keepdim=True)
    re_wav = taf.resample(segmented_audio, sample_rate, tgt_sample_rate)
    torchaudio.save(
        tgt_af,
        re_wav,
        tgt_sample_rate,
        bits_per_sample=bits_per_sample
    )

    duration = re_wav.shape[1] / tgt_sample_rate
    return duration


def timestamp_to_seconds(timestamp: str) -> float:
    """Convert HH:MM:SS.sss timestamp to seconds."""
    hours, minutes, seconds = map(float, timestamp.split(":"))
    return hours * 3600 + minutes * 60 + seconds


if __name__ == "__main__":
    BATCH = "batch_2"
    SAMPLE_RATE = "24k"

    os.makedirs(f"/vast/audio/data/tts/AR-AE/{BATCH}", exist_ok=True)
    with open("/vast/mohamed.bouaziz/data/tts-emirati/recipe/final_jsonl/tmp/naseem_1/split2_naseem2.manifest") as src_m:
        with open(f"/manifest.json", "w") as tgt_m:
            for line in tqdm(src_m):
                try:
                    src_jd = json.loads(line.strip())
                    src_af = src_jd["source"]
                    fid = src_jd["id"]
                    annotations = src_jd["annotation"]
                    batch_1_path = f"/vast/audio/data/tts/AR-AE/batch_1/wav/{SAMPLE_RATE}/{fid}"
                    if os.path.exists(batch_1_path):
                        continue
                    tgt_path = f"/{fid}"
                    os.makedirs(tgt_path, exist_ok=True)

                    wav, sr = torchaudio.load(src_af)

                    for idx, annotation in enumerate(annotations):
                        annotation_properties = annotation["property"]
                        if annotation_properties['@modelarts:attributes:lid'] == "ar-AE":
                            text = annotation_properties['@modelarts:content'].strip()
                            start = timestamp_to_seconds(annotation_properties['@modelarts:start_time'])
                            end = timestamp_to_seconds(annotation_properties['@modelarts:end_time'])
                            tgt_af = os.path.join(tgt_path, f"{fid}-{idx}.wav")
                            duration = segment_audio_by_time(wav, sr, start, end, tgt_af, tgt_sample_rate=24000,
                                                             bits_per_sample=32)

                            jd = {"audio_filepath": tgt_af, "text": text, "u_fid": fid, "duration": duration,
                                  "source": src_af}
                            json.dump(jd, tgt_m)
                            tgt_m.write("\n")
                except Exception as e:
                    _LOG.error(e)