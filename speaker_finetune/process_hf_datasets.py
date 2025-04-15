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
import locale

# Set locale for UTF-8 support
locale.getpreferredencoding = lambda: "UTF-8"

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
        
        # Normalize audio if needed
        if waveform.abs().max() > 1.0:
            waveform = waveform / waveform.abs().max()
        
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

def add_codes(example):
    """Process a single example to add SNAC codes"""
    try:
        # Get audio data
        audio_data = example.get("audio")
        if audio_data is None:
            logger.debug("No audio data found")
            return example
            
        # Handle different audio data formats
        if isinstance(audio_data, dict):
            # Standard HF audio format
            audio_array = audio_data.get("array")
            sample_rate = audio_data.get("sampling_rate", 24000)
        elif isinstance(audio_data, str):
            # Path to audio file
            logger.debug(f"Audio is a path: {audio_data}")
            try:
                waveform, sample_rate = torchaudio.load(audio_data)
                audio_array = waveform.squeeze().numpy()
            except Exception as e:
                logger.debug(f"Error loading audio file: {str(e)}")
                return example
        else:
            logger.debug(f"Unsupported audio format: {type(audio_data)}")
            return example
            
        if audio_array is None:
            logger.debug("No audio array found")
            return example
            
        # Check duration
        duration = len(audio_array) / sample_rate
        min_duration = config.get("min_duration", 0.1)
        max_duration = config.get("max_duration", 30.0)
        if not (min_duration <= duration <= max_duration):
            logger.debug(f"Duration {duration}s outside range {min_duration}-{max_duration}s")
            return example
            
        # Get SNAC codes
        audio_tokens_start = config.get("audio_tokens_start", 128266)
        codes_list = tokenise_audio(audio_array, sample_rate, audio_tokens_start)
        if codes_list is None or len(codes_list) == 0:
            logger.debug("Failed to get SNAC codes")
            return example
            
        # Remove duplicate frames
        codes_list = remove_duplicate_frames(codes_list)
        if codes_list is None or len(codes_list) == 0:
            logger.debug("No codes left after removing duplicates")
            return example
            
        example["codes_list"] = codes_list
        return example
    except Exception as e:
        logger.debug(f"Error processing example: {str(e)}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
        example["codes_list"] = None
        return example

def create_input_ids(example):
    """Create input IDs for training from processed example"""
    try:
        # Initialize fields with empty lists as default
        example["input_ids"] = []
        example["labels"] = []
        example["attention_mask"] = []
        
        if example.get("codes_list") is None:
            return example
            
        # Get text from the example
        text = example.get("text", "")
        if not text:
            return example
        
        # Get speaker information
        speaker = None
        for field in ["source", "speaker", "voice_name", "speaker_name"]:
            if field in example and example[field]:
                speaker = example[field]
                break
                
        # Format text with speaker information for multi-speaker training
        if speaker:
            text_prompt = f"{speaker}: {text}"
        else:
            text_prompt = text
        
        # Special tokens
        start_of_human = config.get("start_of_human", 128259)
        end_of_human = config.get("end_of_human", 128260)
        start_of_ai = config.get("start_of_ai", 128261)
        start_of_speech = config.get("start_of_speech", 128257)
        end_of_speech = config.get("end_of_speech", 128258)
        end_of_ai = config.get("end_of_ai", 128262)
        end_of_text = config.get("end_of_text", 128009)
        
        # Encode text
        text_ids = tokenizer.encode(text_prompt, add_special_tokens=True)
        text_ids.append(end_of_text)
        
        # Create sequence
        input_ids = (
            [start_of_human]
            + text_ids
            + [end_of_human]
            + [start_of_ai]
            + [start_of_speech]
            + example["codes_list"]
            + [end_of_speech]
            + [end_of_ai]
        )
        
        example["input_ids"] = input_ids
        example["labels"] = input_ids
        example["attention_mask"] = [1] * len(input_ids)
        
        return example
    except Exception as e:
        logger.debug(f"Error creating input IDs: {str(e)}")
        # Ensure these fields exist even if there's an error
        if "input_ids" not in example:
            example["input_ids"] = []
        if "labels" not in example:
            example["labels"] = []
        if "attention_mask" not in example:
            example["attention_mask"] = []
        return example

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
        
        # Initialize model and tokenizer
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
        
        # Process each dataset
        all_processed_datasets = []
        
        for dataset_idx, dataset_name in enumerate(dataset_names):
            logger.info(f"Processing dataset {dataset_idx+1}/{len(dataset_names)}: {dataset_name}")
            
            # Load dataset
            try:
                dataset = load_dataset(dataset_name, split=split)
                logger.info(f"Successfully loaded {len(dataset)} examples from {dataset_name}")
                
                # Print sample for debugging
                if len(dataset) > 0:
                    sample = dataset[0]
                    logger.info(f"Sample from {dataset_name}:")
                    logger.info(f"  Has 'text': {('text' in sample)}")
                    if 'text' in sample:
                        logger.info(f"  Text sample: {sample['text'][:50]}...")
                    logger.info(f"  Has 'audio': {('audio' in sample)}")
                    if 'audio' in sample and isinstance(sample['audio'], dict):
                        logger.info(f"  Audio sample rate: {sample['audio'].get('sampling_rate')}")
                        logger.info(f"  Audio duration: {len(sample['audio'].get('array', [])) / sample['audio'].get('sampling_rate', 1):.2f}s")
                    
                    # Check for speaker fields
                    for field in ["source", "speaker", "voice_name", "speaker_name"]:
                        logger.info(f"  Has '{field}': {field in sample}")
                    
                    # Log all available keys
                    logger.info(f"  Available keys: {list(sample.keys())}")
            except Exception as e:
                logger.error(f"Error loading dataset {dataset_name}: {str(e)}")
                continue
            
            # Process in smaller chunks to avoid memory issues
            processed_chunks = []
            
            # Determine chunk size based on dataset size
            chunk_size = min(batch_size, len(dataset))
            num_chunks = (len(dataset) + chunk_size - 1) // chunk_size
            
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min((chunk_idx + 1) * chunk_size, len(dataset))
                
                logger.info(f"Processing chunk {chunk_idx+1}/{num_chunks} ({start_idx}-{end_idx}) from {dataset_name}")
                
                # Extract chunk
                chunk = dataset.select(range(start_idx, end_idx))
                
                # Inside prepare_datasets_from_hf function, replace the processing chunk section with:
                
                try:
                    # Add SNAC codes
                    logger.info(f"Adding SNAC codes to chunk with {len(chunk)} examples")
                    processed_chunk = chunk.map(
                        add_codes,
                        desc="Adding SNAC codes",
                        load_from_cache_file=False  # Disable caching for better error handling
                    )
                    
                    # Verify all examples have codes_list field (even if None)
                    if not all("codes_list" in example for example in processed_chunk):
                        logger.warning("Some examples are missing codes_list field, adding it")
                        processed_chunk = processed_chunk.map(
                            lambda x: {**x, "codes_list": x.get("codes_list", None)},
                            desc="Ensuring codes_list field exists"
                        )
                    
                    # Filter out examples without valid codes
                    processed_chunk = processed_chunk.filter(
                        lambda x: x.get("codes_list") is not None and len(x.get("codes_list", [])) > 0,
                        desc="Filtering valid examples"
                    )
                    
                    if len(processed_chunk) == 0:
                        logger.warning(f"No valid examples with SNAC codes in chunk {chunk_idx+1}")
                        continue
                    
                    # Create input IDs
                    logger.info(f"Creating input IDs for {len(processed_chunk)} examples")
                    processed_chunk = processed_chunk.map(
                        create_input_ids,
                        desc="Creating input IDs",
                        load_from_cache_file=False  # Disable caching for better error handling
                    )
                    
                    # Filter out examples without valid input IDs
                    processed_chunk = processed_chunk.filter(
                        lambda x: "input_ids" in x and len(x.get("input_ids", [])) > 0,
                        desc="Filtering examples with valid input IDs"
                    )
                    
                    if len(processed_chunk) > 0:
                        # Keep only necessary columns to reduce memory usage
                        columns_to_keep = ["input_ids", "labels", "attention_mask"]
                        processed_chunk = processed_chunk.remove_columns(
                            [col for col in processed_chunk.column_names if col not in columns_to_keep]
                        )
                        
                        processed_chunks.append(processed_chunk)
                        logger.info(f"Successfully processed {len(processed_chunk)} examples from chunk {chunk_idx+1}")
                    else:
                        logger.warning(f"No valid examples with input IDs in chunk {chunk_idx+1}")
                    
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_idx+1}: {str(e)}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    continue
                
                # Clear memory
                gc.collect()
                torch.cuda.empty_cache()
            
            # Combine all chunks from this dataset
            if processed_chunks:
                try:
                    dataset_processed = concatenate_datasets(processed_chunks)
                    logger.info(f"Combined {len(dataset_processed)} examples from {len(processed_chunks)} chunks for {dataset_name}")
                    all_processed_datasets.append(dataset_processed)
                except Exception as e:
                    logger.error(f"Error combining chunks for {dataset_name}: {str(e)}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Combine all datasets
        if all_processed_datasets:
            try:
                logger.info(f"Combining {len(all_processed_datasets)} processed datasets")
                final_dataset = concatenate_datasets(all_processed_datasets)
                
                logger.info(f"Saving final dataset with {len(final_dataset)} examples to {output_dir}")
                final_dataset.save_to_disk(output_dir)
                
                # Push to hub if requested
                if push_to_hub and hub_name:
                    logger.info(f"Pushing dataset to hub: {hub_name}")
                    final_dataset.push_to_hub(hub_name)
                
                return final_dataset
            except Exception as e:
                logger.error(f"Error combining datasets: {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                
                # Try to save individual datasets if combining fails
                logger.info("Attempting to save individual datasets instead")
                for i, dataset in enumerate(all_processed_datasets):
                    try:
                        dataset_dir = os.path.join(output_dir, f"dataset_{i}")
                        os.makedirs(dataset_dir, exist_ok=True)
                        dataset.save_to_disk(dataset_dir)
                        logger.info(f"Saved dataset {i} with {len(dataset)} examples to {dataset_dir}")
                    except Exception as e2:
                        logger.error(f"Error saving dataset {i}: {str(e2)}")
        else:
            logger.warning("No data was successfully processed. Creating an empty dataset.")
            empty_dataset = Dataset.from_dict({
                "input_ids": [],
                "labels": [],
                "attention_mask": []
            })
            empty_dataset.save_to_disk(output_dir)
            return empty_dataset
            
    except Exception as e:
        logger.error(f"Error processing datasets: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
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
    
    # Example config.yaml content
    if not os.path.exists(args.config):
        logger.info(f"Config file {args.config} not found, creating default config")
        default_config = {
            "snac_model": "hubertsiuzdak/snac_24khz",
            "tokenizer_name": "unsloth/orpheus-3b-0.1-ft-unsloth-bnb-4bit",
            "min_duration": 0.1,
            "max_duration": 30.0,
            "audio_tokens_start": 128266,
            "start_of_human": 128259,
            "end_of_human": 128260,
            "start_of_ai": 128261,
            "start_of_speech": 128257,
            "end_of_speech": 128258,
            "end_of_ai": 128262,
            "end_of_text": 128009,
            "pad_token": 128264
        }
        with open(args.config, "w") as f:
            yaml.dump(default_config, f)
    
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