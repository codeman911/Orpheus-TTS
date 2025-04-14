import os
import torch
import torchaudio
import numpy as np
from datasets import load_dataset, Dataset, concatenate_datasets
import yaml
from snac import SNAC
import torchaudio.transforms as T
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse
import multiprocessing as mp
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

        # Always use cuda:0 for encoding
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
    """Process a single audio-text pair with speaker information from HF dataset"""
    global config, tokenizer
    
    try:
        # Get audio data
        audio_data = item.get("audio")
        if not audio_data:
            logger.debug("Skipping item with no audio data")
            return None
            
        # Extract waveform and sample rate
        waveform = torch.tensor(audio_data.get("array"), dtype=torch.float32)
        sample_rate = audio_data.get("sampling_rate", 24000)
        
        # Check duration
        duration = len(waveform) / sample_rate
        min_duration = config.get("min_duration", 0.1)
        max_duration = config.get("max_duration", 30.0)
        if not (min_duration <= duration <= max_duration):
            return None
            
        # Ensure mono audio
        if len(waveform.shape) > 1 and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        elif len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
            
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
        
        # Get speaker information - use speaker, voice_name, speaker_name, or a default value
        # Special case for MrDragonFox/Elise dataset
        dataset_name = item.get("_dataset_name", "")
        if dataset_name == "MrDragonFox/Elise":
            speaker = "Elise"
        else:
            speaker = item.get("speaker", item.get("voice_name", item.get("speaker_name", "unknown_speaker")))
        
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
        logger.debug(f"Error processing item: {str(e)}")
        return None

def process_chunk_parallel(items_chunk, num_workers):
    """Process a chunk of items in parallel using ThreadPoolExecutor"""
    if num_workers is None or num_workers <= 1:
        # Process sequentially
        processed_data = []
        skipped_reasons = {"audio": 0, "duration": 0, "codes": 0, "text": 0, "other": 0}
        
        for item in tqdm(items_chunk):
            # Add debug logging to track why items are being skipped
            try:
                if not item.get("audio"):
                    skipped_reasons["audio"] += 1
                    continue
                    
                waveform = torch.tensor(item["audio"].get("array"), dtype=torch.float32)
                sample_rate = item["audio"].get("sampling_rate", 24000)
                duration = len(waveform) / sample_rate
                min_duration = config.get("min_duration", 0.1)
                max_duration = config.get("max_duration", 30.0)
                
                if not (min_duration <= duration <= max_duration):
                    skipped_reasons["duration"] += 1
                    continue
                    
                if not item.get("text"):
                    skipped_reasons["text"] += 1
                    continue
                
                result = process_item(item)
                if result:
                    processed_data.append(result)
                else:
                    skipped_reasons["codes"] += 1
            except Exception as e:
                logger.debug(f"Error in processing: {str(e)}")
                skipped_reasons["other"] += 1
        
        logger.info(f"Skipped items breakdown: {skipped_reasons}")
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
    
    logger.info(f"Processed {len(processed_data)} out of {len(items_chunk)} items")
    return processed_data

def load_hf_dataset(dataset_name, split="train"):
    """Load a dataset from Hugging Face Hub"""
    logger.info(f"Loading dataset: {dataset_name}, split: {split}")
    try:
        dataset = load_dataset(dataset_name, split=split)
        logger.info(f"Successfully loaded {len(dataset)} examples from {dataset_name}")
        
        # Log one sample to check fields
        if len(dataset) > 0:
            sample = dataset[0]
            logger.info(f"Sample from {dataset_name}:")
            logger.info(f"  Has 'text': {sample.get('text') is not None}")
            if sample.get('text'):
                logger.info(f"  Text sample: {sample.get('text')[:50]}...")
            
            logger.info(f"  Has 'audio': {sample.get('audio') is not None}")
            if sample.get('audio'):
                audio_data = sample.get('audio')
                logger.info(f"  Audio sample rate: {audio_data.get('sampling_rate')}")
                logger.info(f"  Audio duration: {len(audio_data.get('array', [])) / audio_data.get('sampling_rate', 1):.2f}s")
            
            logger.info(f"  Has 'speaker': {sample.get('speaker') is not None}")
            logger.info(f"  Has 'voice_name': {sample.get('voice_name') is not None}")
            speaker = sample.get('speaker', sample.get('voice_name', 'unknown_speaker'))
            logger.info(f"  Speaker value: {speaker}")
            
            # Log all available keys in the sample
            logger.info(f"  Available keys: {list(sample.keys())}")
        
        return dataset
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_name}: {str(e)}")
        return None

def prepare_datasets_from_hf(dataset_names, output_dir, config_path="config.yaml", 
                            push_to_hub=False, hub_name=None, num_workers=None, 
                            batch_size=1000, split="train"):
    """Convert Hugging Face datasets to a processed dataset with SNAC tokenization."""
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
        
        logger.info(f"Loading tokenizer: {config.get('tokenizer_name')}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                config.get('tokenizer_name'),
                pad_token_id=config.get("pad_token", 128264)
            )
        except Exception as e:
            logger.error(f"Error loading tokenizer: {str(e)}")
            raise
        
        # Process in chunks
        total_processed = 0
        part_counter = 0
        final_dataset = None  # Initialize final_dataset to None
        
        # Determine optimal number of workers if not specified
        if num_workers is None:
            num_workers = min(os.cpu_count(), 100)  # Default to reasonable number
        
        logger.info(f"Using {num_workers} workers for processing")
        
        # Process each dataset in the list
        for dataset_idx, dataset_name in enumerate(dataset_names):
            logger.info(f"Processing dataset {dataset_idx+1}/{len(dataset_names)}: {dataset_name}")
            
            # Load dataset from Hugging Face
            dataset = load_hf_dataset(dataset_name, split)
            if dataset is None:
                logger.warning(f"Skipping dataset {dataset_name} due to loading error")
                continue
            
            # Add dataset_name to each item for identification in process_item
            dataset = dataset.map(lambda x: {"_dataset_name": dataset_name})
                
            # Process dataset in chunks
            for i in range(0, len(dataset), batch_size):
                chunk_start = i
                chunk_end = min(i + batch_size, len(dataset))
                items_chunk = dataset[chunk_start:chunk_end]
                
                logger.info(f"Processing chunk {i//batch_size + 1}/{(len(dataset) + batch_size - 1)//batch_size} with {len(items_chunk)} items")
                
                # Process items in parallel or sequentially
                processed_data = process_chunk_parallel(items_chunk, num_workers)
                
                if processed_data:
                    # Save this chunk as a dataset part
                    part_counter += 1
                    part_path = os.path.join(output_dir, f"part_{part_counter}")
                    logger.info(f"Saving part {part_counter} with {len(processed_data)} examples to {part_path}")
                    
                    part_dataset = Dataset.from_list(processed_data)
                    part_dataset.save_to_disk(part_path)
                    
                    total_processed += len(processed_data)
                    
                    # Clear memory
                    del processed_data
                    del part_dataset
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
        else:
            # Create an empty dataset if no data was processed
            logger.warning("No data was successfully processed. Creating an empty dataset.")
            final_dataset = Dataset.from_dict({
                "input_ids": [],
                "labels": [],
                "attention_mask": []
            })
            final_dataset.save_to_disk(output_dir)
        
        logger.info(f"Successfully processed {total_processed} examples from {len(dataset_names)} datasets")
        
        # Push to hub if requested
        if push_to_hub and hub_name and final_dataset is not None:
            logger.info(f"Pushing dataset to hub: {hub_name}")
            final_dataset.push_to_hub(hub_name)
        
        return final_dataset
        
    except Exception as e:
        logger.error(f"Error processing datasets: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare text-speech dataset from Hugging Face datasets")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--output", type=str, required=True, help="Output directory for processed dataset")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for processing")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.info("Debug logging enabled")
    
    # Hardcoded list of datasets
    dataset_names = [
        "MrDragonFox/Elise",
        "Jinsaryko/Eden",
        "Jinsaryko/Ceylia",
        "Jinsaryko/Hope",
        "Jinsaryko/Alexis",
        "Jinsaryko/Aerith",
        "Jinsaryko/Raina",
        "Jinsaryko/Gem",
        "Jinsaryko/Infinity",
        "Jinsaryko/Jade",
        "Jinsaryko/Crystal",
    ]
    
    prepare_datasets_from_hf(
        dataset_names,
        args.output,
        args.config,
        False,  # Never push to hub
        None,   # No hub name needed
        args.workers,
        args.batch_size,
        args.split
    )