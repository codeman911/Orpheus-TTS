import os
import numpy as np
import torch
import torchaudio.transforms as T
from datasets import load_dataset
from snac import SNAC
from transformers import AutoTokenizer
import yaml
import logging

# --- Config ---
DATASET_NAME = "v1kram/expresso_nx_dataset"
SPLIT = "train"
OUTPUT_PATH = "output_exp"
CONFIG_PATH = "config.yaml"
NUM_PROC = 1  # Set to os.cpu_count() for max parallelism

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Load config ---
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# --- Load SNAC model and tokenizer ---
model = SNAC.from_pretrained(config.get("snac_model", "hubertsiuzdak/snac_24khz")).eval().to("cpu")
tokenizer = AutoTokenizer.from_pretrained(
    config.get("tokenizer_name"),
    pad_token_id=config.get("pad_token", 128264)
)

# --- Tokenization helper ---
def tokenise_audio(waveform, sample_rate=24000, audio_tokens_start=128266):
    try:
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform).unsqueeze(0)
        waveform = waveform.to(dtype=torch.float32)
        if sample_rate != 24000:
            resample_transform = T.Resample(orig_freq=sample_rate, new_freq=24000)
            waveform = resample_transform(waveform)
        waveform = waveform.unsqueeze(0).to("cpu")
        with torch.inference_mode():
            codes = model.encode(waveform)
        all_codes = []
        for i in range(codes[0].shape[1]):
            all_codes.append(codes[0][0][i].item() + audio_tokens_start)
            all_codes.append(codes[1][0][2*i].item() + audio_tokens_start + 4096)
            all_codes.append(codes[2][0][4*i].item() + audio_tokens_start + (2*4096))
            all_codes.append(codes[2][0][(4*i)+1].item() + audio_tokens_start + (3*4096))
            all_codes.append(codes[1][0][(2*i)+1].item() + audio_tokens_start + (4*4096))
            all_codes.append(codes[2][0][(4*i)+2].item() + audio_tokens_start + (5*4096))
            all_codes.append(codes[2][0][(4*i)+3].item() + audio_tokens_start + (6*4096))
        return all_codes
    except Exception as e:
        logger.warning(f"Error in tokenise_audio: {str(e)}")
        return None

def remove_duplicate_frames(codes_list):
    if codes_list is None or len(codes_list) == 0:
        return None
    if len(codes_list) % 7 != 0:
        codes_list = codes_list[:-(len(codes_list) % 7)]
    if len(codes_list) == 0:
        return None
    result = codes_list[:7]
    for i in range(7, len(codes_list), 7):
        if i+6 >= len(codes_list):
            break
        current_first = codes_list[i]
        previous_first = result[-7]
        if current_first != previous_first:
            result.extend(codes_list[i:i+7])
    if len(result) < 7:
        return None
    return result

# --- Main processing function ---
def process_example(example):
    try:
        audio_data = example["audio"]
        waveform = torch.tensor(audio_data["array"], dtype=torch.float32)
        sample_rate = audio_data["sampling_rate"]
        duration = len(waveform) / sample_rate
        if not (config.get("min_duration", 0.1) <= duration <= config.get("max_duration", 30.0)):
            return {}
        codes_list = tokenise_audio(waveform, sample_rate, config.get("audio_tokens_start", 128266))
        codes_list = remove_duplicate_frames(codes_list)
        if not codes_list:
            return {}
        text = example.get("text", "")
        if not text:
            return {}
        speaker = example.get("speaker_name", "unknown_speaker")
        text_prompt = f"{speaker}: {text}"
        text_ids = tokenizer.encode(text_prompt, add_special_tokens=True)
        text_ids.append(config.get("end_of_text", 128009))
        input_ids = (
            [config.get("start_of_human", 128259)]
            + text_ids
            + [config.get("end_of_human", 128260)]
            + [config.get("start_of_ai", 128261)]
            + [config.get("start_of_speech", 128257)]
            + codes_list
            + [config.get("end_of_speech", 128258)]
            + [config.get("end_of_ai", 128262)]
        )
        return {
            "input_ids": input_ids,
            "labels": input_ids,
            "attention_mask": [1] * len(input_ids)
        }
    except Exception as e:
        logger.warning(f"Error processing example: {str(e)}")
        return {}

# --- Load dataset ---
ds = load_dataset(DATASET_NAME, split=SPLIT)

# --- Map processing function ---
processed_ds = ds.map(
    process_example,
    num_proc=NUM_PROC,
    remove_columns=ds.column_names,
    desc="Processing dataset"
)

# --- Filter out empty examples ---
processed_ds = processed_ds.filter(lambda x: len(x["input_ids"]) > 0)

# --- Save processed dataset ---
processed_ds.save_to_disk(OUTPUT_PATH)
print(f"✅ Processed dataset saved to {OUTPUT_PATH}")
