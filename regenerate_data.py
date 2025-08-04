import os
import json
import torch
import torchaudio
import numpy as np
from datasets import load_from_disk
from snac import SNAC
import logging
from pathlib import Path
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global SNAC model
model = None

def initialize_snac():
    """Initialize SNAC model for decoding"""
    global model
    if model is None:
        logger.info("Loading SNAC model...")
        model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        logger.info(f"SNAC model loaded on {device}")
    return model

def extract_audio_codes_from_tokens(tokens, audio_tokens_start=128266):
    """Extract audio codes from token sequence"""
    audio_tokens = []
    for token in tokens:
        if token >= audio_tokens_start and token < audio_tokens_start + 7 * 4096:
            # This is an audio token
            layer = (token - audio_tokens_start) // 4096
            code = (token - audio_tokens_start) % 4096
            audio_tokens.append((layer, code))
    
    # Reconstruct the SNAC format
    if len(audio_tokens) % 7 != 0:
        logger.warning(f"Audio tokens length {len(audio_tokens)} not divisible by 7, truncating")
        audio_tokens = audio_tokens[:-(len(audio_tokens) % 7)]
    
    if len(audio_tokens) == 0:
        return None
    
    # Group by frames
    frames = []
    for i in range(0, len(audio_tokens), 7):
        frame = audio_tokens[i:i+7]
        if len(frame) == 7:
            frames.append(frame)
    
    # Reconstruct SNAC codes
    codes_0 = []
    codes_1 = []
    codes_2 = []
    
    for frame in frames:
        # Frame structure: [layer0, layer1, layer2, layer3, layer4, layer5, layer6]
        # Based on the encoding pattern in zero_shot_datasets.py
        codes_0.append(frame[0][1])  # First token is layer 0
        codes_1.extend([frame[1][1], frame[4][1]])  # Second and fifth tokens are layer 1
        codes_2.extend([frame[2][1], frame[3][1], frame[5][1], frame[6][1]])  # Rest are layer 2
    
    return [torch.tensor([codes_0]), torch.tensor([codes_1]), torch.tensor([codes_2])]

def decode_snac_to_audio(codes):
    """Decode SNAC codes to audio waveform"""
    if codes is None or len(codes) != 3:
        return None
    
    try:
        model = initialize_snac()
        
        # Ensure codes are on the right device
        device = next(model.parameters()).device
        codes = [c.to(device) if isinstance(c, torch.Tensor) else torch.tensor(c, device=device) for c in codes]
        
        with torch.inference_mode():
            audio_hat = model.decode(codes)
        
        # Convert to numpy array
        audio_np = audio_hat.squeeze().cpu().numpy()
        return audio_np
    except Exception as e:
        logger.error(f"Error decoding SNAC codes: {e}")
        return None

def save_audio(audio, output_path, sample_rate=24000):
    """Save audio to file"""
    if audio is None:
        return False
    
    try:
        # Ensure audio is in the right format
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
        else:
            audio_tensor = audio.float()
        
        # Ensure audio is 1D
        if len(audio_tensor.shape) > 1:
            audio_tensor = audio_tensor.squeeze()
        
        torchaudio.save(output_path, audio_tensor.unsqueeze(0), sample_rate)
        logger.info(f"Saved audio to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving audio to {output_path}: {e}")
        return False

def save_text(text, output_path):
    """Save text to file"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        logger.info(f"Saved text to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving text to {output_path}: {e}")
        return False

def process_dataset(dataset_path, output_dir, num_samples=None):
    """Process dataset and regenerate audio from SNAC codes"""
    logger.info(f"Loading dataset from {dataset_path}")
    dataset = load_from_disk(dataset_path)
    
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    # Create single output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize metadata
    metadata = {
        "dataset_info": {
            "total_samples": 0,
            "speakers": ["default_spk"],
            "total_duration": 0.0,
            "avg_duration": 0.0
        },
        "samples": []
    }
    
    # Process samples
    processed_count = 0
    for idx in range(len(dataset)):
        sample = dataset[idx]
        logger.info(f"Processing sample {idx + 1}/{len(dataset)}")
        
        # Extract reference and target audio codes from input_ids and labels
        input_ids = sample['input_ids']
        labels = sample['labels']
        
        # Token IDs from zero_shot_datasets.py configuration
        start_of_human = 128259
        end_of_human = 128260  
        start_of_ai = 128261
        start_of_speech = 128257
        end_of_speech = 128258
        end_of_text = 128009
        
        # Extract reference audio codes (between start_of_speech and end_of_speech in input_ids)
        try:
            start_speech_idx = input_ids.index(start_of_speech)
            end_speech_idx = input_ids.index(end_of_speech)
            reference_codes = input_ids[start_speech_idx + 1:end_speech_idx]
        except ValueError:
            logger.warning(f"Could not find reference speech boundaries in sample {idx}")
            continue
        
        # Extract target audio codes (between start_of_speech and end_of_speech in labels after input_ids)
        try:
            # Find the AI response part in labels
            ai_response_start = len(input_ids)
            if len(labels) > ai_response_start:
                ai_response = labels[ai_response_start:]
                start_speech_target = ai_response.index(start_of_speech)
                end_speech_target = ai_response.index(end_of_speech)
                target_codes = ai_response[start_speech_target + 1:end_speech_target]
            else:
                logger.warning(f"No AI response found in sample {idx}")
                continue
        except ValueError:
            logger.warning(f"Could not find target speech boundaries in sample {idx}")
            continue
        
        # Create unique ID for this sample
        sample_id = f"sample_{idx:08d}"
        
        # Regenerate reference audio
        ref_codes = extract_audio_codes_from_tokens(reference_codes)
        if ref_codes:
            ref_audio = decode_snac_to_audio(ref_codes)
            ref_audio_path = os.path.join(output_dir, f"{sample_id}_reference.wav")
            save_audio(ref_audio, ref_audio_path)
        
        # Regenerate target audio
        target_codes_list = extract_audio_codes_from_tokens(target_codes)
        if target_codes_list:
            target_audio = decode_snac_to_audio(target_codes_list)
            target_audio_path = os.path.join(output_dir, f"{sample_id}_target.wav")
            save_audio(target_audio, target_audio_path)
        
        # Calculate duration (approximate based on audio length)
        duration = len(target_audio) / 24000 if target_audio is not None else 0.0
        
        # Extract and save reference text
        ref_text = ""
        try:
            # Extract reference text from input_ids
            first_human_start = input_ids.index(start_of_human)
            first_human_end = input_ids.index(end_of_human, first_human_start)
            
            # Extract text tokens between start_of_human and end_of_human
            # Skip the end_of_text token before end_of_human
            ref_text_tokens = []
            for token in input_ids[first_human_start + 1:first_human_end]:
                if token != end_of_text:
                    ref_text_tokens.append(token)
            
            if ref_text_tokens:
                # Decode text tokens
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained("v1kram/zer_v3")
                ref_text = tokenizer.decode(ref_text_tokens, skip_special_tokens=True)
                
                ref_text_path = os.path.join(output_dir, f"{sample_id}_reference.txt")
                save_text(ref_text, ref_text_path)
            else:
                logger.warning(f"No reference text found in sample {idx}")
        except (ValueError, IndexError) as e:
            logger.warning(f"Could not extract reference text from sample {idx}: {e}")
        
        # Extract and save target text
        target_text = ""
        try:
            # Find the second human turn
            second_human_start = None
            human_count = 0
            for i, token in enumerate(input_ids):
                if token == start_of_human:
                    human_count += 1
                    if human_count == 2:
                        second_human_start = i
                        break
            
            if second_human_start is not None:
                second_human_end = input_ids.index(end_of_human, second_human_start)
                
                # Extract text tokens between second human turn and end_of_human
                target_text_tokens = []
                for token in input_ids[second_human_start + 1:second_human_end]:
                    if token != end_of_text:
                        target_text_tokens.append(token)
                
                if target_text_tokens:
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained("v1kram/zer_v3")
                    target_text = tokenizer.decode(target_text_tokens, skip_special_tokens=True)
                    target_text_path = os.path.join(output_dir, f"{sample_id}_target.txt")
                    save_text(target_text, target_text_path)
                else:
                    logger.warning(f"No target text found in sample {idx}")
            else:
                logger.warning(f"Could not find second human turn in sample {idx}")
        except (ValueError, IndexError) as e:
            logger.warning(f"Could not extract target text from sample {idx}: {e}")
        
        # Add to metadata
        metadata["samples"].append({
            "id": sample_id,
            "audio_file": f"{sample_id}_target.wav",
            "transcript_file": f"{sample_id}_target.txt",
            "ref_audio_file": f"{sample_id}_reference.wav",
            "ref_transcript": ref_text,
            "duration": duration,
            "speaker_id": "default_spk"
        })
        
        processed_count += 1
    
    # Update metadata
    metadata["dataset_info"]["total_samples"] = processed_count
    if processed_count > 0:
        total_duration = sum(sample["duration"] for sample in metadata["samples"])
        metadata["dataset_info"]["total_duration"] = total_duration
        metadata["dataset_info"]["avg_duration"] = total_duration / processed_count
    
    # Save metadata.json
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Processing complete! Processed {processed_count} samples")
    logger.info(f"Output saved to: {output_dir}")
    logger.info(f"Metadata saved to: {metadata_path}")
    
    return {
        "processed_count": processed_count,
        "output_dir": output_dir,
        "metadata_path": metadata_path
    }

def main():
    parser = argparse.ArgumentParser(description="Regenerate audio from SNAC codes in zero-shot dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory for regenerated files")
    parser.add_argument("--num-samples", type=int, help="Number of samples to process (default: all)")
    
    args = parser.parse_args()
    
    result = process_dataset(args.dataset, args.output, args.num_samples)
    
    print(f"\n=== Processing Summary ===")
    print(f"Processed samples: {result['processed_count']}")
    print(f"Output directory: {result['output_dir']}")
    print(f"Metadata file: {result['metadata_path']}")

if __name__ == "__main__":
    main()

