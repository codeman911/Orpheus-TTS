import json
import os
import torch
import torchaudio
import numpy as np
from datasets import Dataset, concatenate_datasets
import yaml
from snac import SNAC
import torchaudio.transforms as T
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse
import multiprocessing as mp
from itertools import islice
import gc
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor

# Set multiprocessing start method to 'spawn' to avoid CUDA issues
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    # Method already set
    pass

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Global model variables
model = None
tokenizer = None
config = None

def tokenise_audio(waveform, sample_rate=24000, audio_tokens_start=128266):
    """Convert audio waveform to SNAC tokens"""
    global model
    
    try:
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform).unsqueeze(0)
        
        waveform = waveform.to(dtype=torch.float32)
        
        if sample_rate != 24000:
            resample_transform = T.Resample(orig_freq=sample_rate, new_freq=24000)
            waveform = resample_transform(waveform)

        # Always use cuda:0 like in tokenise_speech_dataset.py
        waveform = waveform.unsqueeze(0).to("cuda:0")

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
    """Remove duplicate frames from codes_list"""
    if codes_list is None or len(codes_list) == 0:
        return None
        
    if len(codes_list) % 7 != 0:
        # Truncate to nearest multiple of 7
        codes_list = codes_list[:-(len(codes_list) % 7)]
    
    if len(codes_list) == 0:
        return None
        
    result = codes_list[:7]
    
    for i in range(7, len(codes_list), 7):
        # Check if we have a complete frame
        if i+6 >= len(codes_list):
            break
            
        current_first = codes_list[i]
        previous_first = result[-7]
        
        if current_first != previous_first:
            result.extend(codes_list[i:i+7])
    
    # Final check to ensure we have at least one frame
    if len(result) < 7:
        return None
        
    return result

def process_item(item):
    """Process a single audio-text pair with speaker information"""
    global config, tokenizer
    
    try:
        # Check if file exists
        audio_path = item.get("audio_filepath")
        if not audio_path or not os.path.exists(audio_path):
            logger.debug(f"Skipping non-existent audio file: {audio_path}")
            return None
            
        # Check duration
        min_duration = config.get("min_duration", 0.1)
        max_duration = config.get("max_duration", 30.0)
        if not (min_duration <= item.get("duration", 0) <= max_duration):
            return None
            
        # Load and process audio
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:  # Convert stereo to mono
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Get SNAC codes
        audio_tokens_start = config.get("audio_tokens_start", 128266)
        codes_list = tokenise_audio(waveform, sample_rate, audio_tokens_start)
        if codes_list is None or len(codes_list) == 0:
            return None
            
        # Remove duplicate frames
        codes_list = remove_duplicate_frames(codes_list)
        if codes_list is None or len(codes_list) == 0:
            return None
            
        # Get text from the example
        text = item.get("text", "")
        if not text:
            return None
        
        # Get speaker information - use speaker, voice_name, or a default value
        speaker = item.get("speaker", item.get("voice_name", "unknown_speaker"))
        
        # Format text with speaker information for multi-speaker training
        text_prompt = f"{speaker}: {text}"
        
        # Encode text with speaker information
        text_ids = tokenizer.encode(text_prompt, add_special_tokens=True)
        text_ids.append(config.get("end_of_text", 128009))
        
        # Create sequence with proper token IDs from config
        start_of_human = config.get("start_of_human", 128259)
        end_of_human = config.get("end_of_human", 128260)
        start_of_ai = config.get("start_of_ai", 128261)
        start_of_speech = config.get("start_of_speech", 128257)
        end_of_speech = config.get("end_of_speech", 128258)
        end_of_ai = config.get("end_of_ai", 128262)
        
        input_ids = (
            [start_of_human]
            + text_ids
            + [end_of_human]
            + [start_of_ai]
            + [start_of_speech]
            + codes_list
            + [end_of_speech]
            + [end_of_ai]
        )
        
        return {
            "input_ids": input_ids,
            "labels": input_ids,
            "attention_mask": [1] * len(input_ids)
        }
    except Exception as e:
        logger.debug(f"Error processing file {item.get('audio_filepath', 'unknown')}: {str(e)}")
        return None

def read_manifest_in_chunks(manifest_path, chunk_size=100000):
    """Read manifest file in chunks to avoid loading everything into memory"""
    with open(manifest_path, 'r') as f:
        chunk = []
        skipped_files = 0
        for i, line in enumerate(f):
            if line.strip():
                try:
                    item = json.loads(line)
                    # Pre-check if audio file exists to avoid loading invalid items
                    audio_path = item.get("audio_filepath")
                    if not audio_path or not os.path.exists(audio_path):
                        skipped_files += 1
                        if skipped_files % 1000 == 0:
                            logger.warning(f"Skipped {skipped_files} non-existent audio files so far")
                        continue
                        
                    chunk.append(item)
                    if len(chunk) >= chunk_size:
                        logger.info(f"Yielding chunk with {len(chunk)} valid items (skipped {skipped_files} invalid files)")
                        yield chunk
                        chunk = []
                except json.JSONDecodeError:
                    logger.debug(f"Error parsing JSON at line {i+1}")
        
        if chunk:  # Don't forget the last chunk
            logger.info(f"Yielding final chunk with {len(chunk)} valid items (skipped {skipped_files} invalid files in total)")
            yield chunk

def process_chunk_parallel(items_chunk, num_workers):
    """Process a chunk of items in parallel using ThreadPoolExecutor"""
    if num_workers is None or num_workers <= 1:
        # Process sequentially
        processed_data = []
        for item in tqdm(items_chunk):
            result = process_item(item)
            if result:
                processed_data.append(result)
        return processed_data
    
    # Process in parallel with optimized chunk size:
    processed_data = []
    # Calculate optimal chunk size based on number of items and workers
    chunk_size = max(1, len(items_chunk) // (num_workers * 4))
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Use map with chunking for better load balancing
        results = list(tqdm(
            executor.map(process_item, items_chunk, chunksize=chunk_size),
            total=len(items_chunk),
            desc="Processing items"
        ))
        processed_data = [r for r in results if r is not None]
    
    return processed_data

def prepare_dataset_from_manifest(manifest_path, output_dir, config_path="config.yaml", push_to_hub=False, hub_name=None, num_workers=None, batch_size=1000):
    """Convert a local manifest file to a dataset with SNAC tokenization and speaker information."""
    global config, model, tokenizer
    
    try:
        # Load config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Enable CUDA optimization
        torch.backends.cudnn.benchmark = True
        
        # Initialize model and tokenizer in main process only
        logger.info(f"Loading SNAC model: {config.get('snac_model', 'hubertsiuzdak/snac_24khz')}")
        try:
            model = SNAC.from_pretrained(config.get('snac_model', 'hubertsiuzdak/snac_24khz')).eval().to("cuda:0")
        except Exception as e:
            logger.error(f"Error loading SNAC model: {str(e)}")
            raise
        
        logger.info(f"Loading tokenizer: {config['tokenizer_name']}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                config['tokenizer_name'],
                pad_token_id=config.get("pad_token", 128264)
            )
        except Exception as e:
            logger.error(f"Error loading tokenizer: {str(e)}")
            raise
        
        # Process in chunks
        total_processed = 0
        part_counter = 0
        
        # Determine optimal number of workers if not specified
        if num_workers is None:
            num_workers = min(os.cpu_count(), 100)  # Default to reasonable number
        
        logger.info(f"Using {num_workers} workers for processing")
        
        # Process manifest in chunks
        for chunk_idx, items_chunk in enumerate(read_manifest_in_chunks(manifest_path, batch_size)):
            logger.info(f"Processing chunk {chunk_idx+1} with {len(items_chunk)} items")
            
            # Process items in parallel or sequentially
            processed_data = process_chunk_parallel(items_chunk, num_workers)
            
            if processed_data:
                # Save this chunk as a dataset part
                part_counter += 1
                part_path = os.path.join(output_dir, f"part_{part_counter}")
                logger.info(f"Saving part {part_counter} with {len(processed_data)} examples to {part_path}")
                
                dataset = Dataset.from_list(processed_data)
                dataset.save_to_disk(part_path)
                
                total_processed += len(processed_data)
                
                # Clear memory
                del processed_data
                del dataset
                gc.collect()
                torch.cuda.empty_cache()
        
        # Merge all parts
        if part_counter > 0:
            logger.info(f"Merging {part_counter} dataset parts...")
            parts = [Dataset.load_from_disk(os.path.join(output_dir, f"part_{i}")) 
                    for i in range(1, part_counter + 1)]
            
            final_dataset = concatenate_datasets(parts)
            
            # Keep only necessary columns for training
            columns_to_keep = ["input_ids", "labels", "attention_mask"]
            final_dataset = final_dataset.remove_columns([col for col in final_dataset.column_names if col not in columns_to_keep])
            
            final_dataset.save_to_disk(output_dir)
            
            # Cleanup
            for i in range(1, part_counter + 1):
                shutil.rmtree(os.path.join(output_dir, f"part_{i}"))
        
        logger.info(f"Successfully processed {total_processed} examples")
        
        # Push to hub if requested
        if push_to_hub and hub_name:
            logger.info(f"Pushing dataset to hub: {hub_name}")
            final_dataset.push_to_hub(hub_name)
        
        return final_dataset
        
    except Exception as e:
        logger.error(f"Error processing dataset: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare text-speech dataset for training with speaker information")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--output", type=str, required=True, help="Output directory for processed dataset")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest file")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for processing")
    parser.add_argument("--push-to-hub", action="store_true", help="Push dataset to HuggingFace Hub")
    parser.add_argument("--hub-name", type=str, help="Name to use when pushing to HuggingFace Hub")
    
    args = parser.parse_args()
    
    prepare_dataset_from_manifest(
        args.manifest,
        args.output,
        args.config,
        args.push_to_hub,
        args.hub_name,
        args.workers,
        args.batch_size
    )