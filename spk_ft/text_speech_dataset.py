import json
import os
import torch
import torchaudio
import numpy as np
from datasets import Dataset
import yaml
from snac import SNAC
import torchaudio.transforms as T
from transformers import AutoTokenizer
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# Global SNAC model to avoid reloading
_GLOBAL_SNAC_MODEL = None

def init_snac_model():
    """Initialize SNAC model once per process"""
    global _GLOBAL_SNAC_MODEL
    if _GLOBAL_SNAC_MODEL is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SNAC model on {device}...")
        _GLOBAL_SNAC_MODEL = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)
    return _GLOBAL_SNAC_MODEL

def tokenise_audio(waveform, sample_rate=24000):
    """Convert audio waveform to SNAC tokens"""
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform).unsqueeze(0)
    
    waveform = waveform.to(dtype=torch.float32)
    
    if sample_rate != 24000:
        resample_transform = T.Resample(orig_freq=sample_rate, new_freq=24000)
        waveform = resample_transform(waveform)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    waveform = waveform.unsqueeze(0).to(device)

    # Use global model
    model = init_snac_model()
    
    with torch.inference_mode():
        codes = model.encode(waveform)

    # Initialize all_codes list before using it
    all_codes = []
    for i in range(codes[0].shape[1]):
        all_codes.append(codes[0][0][i].item()+128266)
        all_codes.append(codes[1][0][2*i].item()+128266+4096)
        all_codes.append(codes[2][0][4*i].item()+128266+(2*4096))
        all_codes.append(codes[2][0][(4*i)+1].item()+128266+(3*4096))
        all_codes.append(codes[1][0][(2*i)+1].item()+128266+(4*4096))
        all_codes.append(codes[2][0][(4*i)+2].item()+128266+(5*4096))
        all_codes.append(codes[2][0][(4*i)+3].item()+128266+(6*4096))

    return all_codes

def remove_duplicate_frames(codes_list):
    """Remove duplicate frames"""
    if len(codes_list) % 7 != 0:
        print("Warning: Code list length not divisible by 7, truncating")
        codes_list = codes_list[:len(codes_list) - (len(codes_list) % 7)]
    
    if len(codes_list) == 0:
        return []
        
    result = codes_list[:7]
    
    for i in range(7, len(codes_list), 7):
        if i+6 >= len(codes_list):
            break
            
        current_first = codes_list[i]
        previous_first = result[-7]
        
        if current_first != previous_first:
            result.extend(codes_list[i:i+7])
    
    return result

def process_item(item, config, tokenizer):
    """Process a single audio item"""
    try:
        # Load and process audio
        waveform, sample_rate = torchaudio.load(item["audio_file_path"])
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Generate SNAC codes
        codes_list = tokenise_audio(waveform, sample_rate)
        codes_list = remove_duplicate_frames(codes_list)
        
        if len(codes_list) == 0:
            return None
        text = item.get("text", "")
        if not text:
            return None
        
        # Get speaker information - use speaker, voice_name, or a default value
        speaker = item.get("speaker_name", item.get("voice_name", "unknown_speaker"))
        
        # Format text with speaker information for multi-speaker training
        text_prompt = f"{speaker}: {text}"
        # print(text_prompt, "=============================")
        # Encode text
        text_ids = tokenizer.encode(text_prompt, add_special_tokens=True)
        text_ids.append(config["end_of_text"])
        
        # Create sequence
        input_ids = (
            [config["start_of_human"]]
            + text_ids
            + [config["end_of_human"]]
            + [config["start_of_ai"]]
            + [config["start_of_speech"]]
            + codes_list
            + [config["end_of_speech"]]
            + [config["end_of_ai"]]
        )
        
        return {
            "input_ids": input_ids,
            "labels": input_ids,
            "attention_mask": [1] * len(input_ids)
        }
        
    except Exception as e:
        print(f"Error processing {item['audio_file_path']}: {str(e)}")
        return None

def convert_manifest_to_dataset(manifest_path, output_dir):
    """Convert a local manifest file to a dataset with SNAC tokenization."""
    try:
        # Load config
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config["model_name"],
            pad_token_id=config["pad_token"]
        )
        
        # Read manifest
        print(f"Reading manifest file: {manifest_path}")
        with open(manifest_path, 'r') as f:
            items = [json.loads(line) for line in f if line.strip()]
        
        print(f"Found {len(items)} items in manifest")
        
        # Filter by duration
        valid_items = [
            item for item in items
            if config["min_duration"] <= item.get("duration", 0) <= config["max_duration"]
            and os.path.exists(item["audio_file_path"])
        ]
        
        print(f"Valid items after duration filtering: {len(valid_items)}")
        
        # Process items in parallel
        num_workers = min(mp.cpu_count(), 8)  # Use up to 8 workers
        # print(f"Processing with {num_workers} workers")
        
        # Initialize the model in the main process first
        init_snac_model()
        
        # Process in batches to avoid memory issues
        batch_size = 100
        processed_data = []
        
        for i in range(0, len(valid_items), batch_size):
            batch = valid_items[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(valid_items) + batch_size - 1)//batch_size}")
            
            # Process batch with single worker if using GPU, multiple workers if CPU
            if torch.cuda.is_available():
                # GPU processing - sequential is often faster due to model transfer overhead
                results = []
                for item in tqdm(batch, desc="Processing audio files"):
                    result = process_item(item, config, tokenizer)
                    if result:
                        results.append(result)
            else:
                # CPU processing - parallel is faster
                with mp.Pool(num_workers, initializer=init_snac_model) as pool:
                    process_fn = partial(process_item, config=config, tokenizer=tokenizer)
                    results = list(tqdm(
                        pool.imap(process_fn, batch),
                        total=len(batch),
                        desc="Processing audio files"
                    ))
                    results = [r for r in results if r is not None]
            
            processed_data.extend(results)
            print(f"Processed {len(processed_data)} items so far")
        
        # Create and save dataset
        dataset = Dataset.from_list(processed_data)
        dataset.save_to_disk(output_dir)
        
        print(f"Dataset saved to {output_dir}")
        print(f"Successfully processed: {len(dataset)}")
        print(f"Skipped: {len(valid_items) - len(processed_data)}")
        
        return dataset
        
    except Exception as e:
        print(f"Error processing dataset: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    
    args = parser.parse_args()
    convert_manifest_to_dataset(args.manifest, args.output)