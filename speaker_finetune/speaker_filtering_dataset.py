import json
import os
import argparse
import logging
import torchaudio
import torch
import numpy as np
import shutil
from tqdm import tqdm
from collections import defaultdict
import random

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def parse_speaker_info(speaker_info_file):
    """Parse speaker info file and return dict of speakers with duration > 1 hour"""
    speakers_data = {}
    
    try:
        with open(speaker_info_file, 'r') as f:
            lines = f.readlines()
            
        # Skip header lines
        start_idx = 0
        for i, line in enumerate(lines):
            if "Speaker Name" in line and "Duration (hours)" in line:
                start_idx = i + 2  # Skip header and separator line
                break
        
        # Parse speaker data
        for line in lines[start_idx:]:
            line = line.strip()
            if not line or '|' not in line:
                continue
                
            parts = line.split('|')
            if len(parts) != 2:
                continue
                
            speaker_name = parts[0].strip()
            try:
                duration = float(parts[1].strip())
                if duration >= 0.01:  # Only keep speakers with >= 1 hour
                    speakers_data[speaker_name] = duration
            except ValueError:
                continue
    
        logger.info(f"Found {len(speakers_data)} speakers with duration >= 1 hour")
        return speakers_data
        
    except Exception as e:
        logger.error(f"Error parsing speaker info file: {str(e)}")
        return {}

def filter_manifest(input_manifest, output_manifest, selected_speakers):
    """Filter manifest to only include selected speakers"""
    try:
        filtered_count = 0
        total_count = 0
        
        with open(input_manifest, 'r') as in_file, open(output_manifest, 'w') as out_file:
            for line in tqdm(in_file, desc="Filtering manifest"):
                total_count += 1
                if not line.strip():
                    continue
                    
                try:
                    item = json.loads(line)
                    
                    # Check if speaker field exists, otherwise try voice_name
                    speaker = item.get("speaker", item.get("voice_name", None))
                    
                    if speaker in selected_speakers:
                        out_file.write(line)
                        filtered_count += 1
                except json.JSONDecodeError:
                    logger.debug(f"Error parsing JSON line: {line[:50]}...")
        
        logger.info(f"Filtered {filtered_count} out of {total_count} entries")
        return filtered_count
        
    except Exception as e:
        logger.error(f"Error filtering manifest: {str(e)}")
        return 0

def extract_speaker_profiles(input_manifest, output_dir, selected_speakers, min_duration=7.0):
    """Extract representative audio samples for each speaker"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # First pass: collect all eligible files by speaker
        speaker_files = defaultdict(list)
        
        with open(input_manifest, 'r') as in_file:
            for line in tqdm(in_file, desc="Finding eligible audio files"):
                if not line.strip():
                    continue
                    
                try:
                    item = json.loads(line)
                    
                    # Check if speaker field exists, otherwise try voice_name
                    speaker = item.get("speaker", item.get("voice_name", None))
                    
                    if speaker in selected_speakers:
                        duration = item.get("duration", 0)
                        audio_path = item.get("audio_filepath")
                        
                        if duration >= min_duration and audio_path and os.path.exists(audio_path):
                            speaker_files[speaker].append((audio_path, duration))
                except json.JSONDecodeError:
                    continue
        
        # Second pass: select and copy files
        for speaker, files in tqdm(speaker_files.items(), desc="Extracting speaker profiles"):
            if not files:
                logger.warning(f"No eligible files found for speaker {speaker}")
                continue
                
            # Sort by duration (descending) and take the longest file
            files.sort(key=lambda x: x[1], reverse=True)
            selected_file, duration = files[0]
            
            # Create output filename with rounded duration
            rounded_duration = round(duration)
            output_filename = f"{speaker}_{rounded_duration}.wav"
            output_path = os.path.join(output_dir, output_filename)
            
            # Copy the file
            try:
                # Load and save with torchaudio to ensure consistent format
                waveform, sample_rate = torchaudio.load(selected_file)
                torchaudio.save(output_path, waveform, sample_rate)
                logger.info(f"Saved profile for {speaker}: {output_path} ({duration:.2f}s)")
            except Exception as e:
                logger.error(f"Error copying file for {speaker}: {str(e)}")
                # Fallback to direct copy if torchaudio fails
                try:
                    shutil.copy(selected_file, output_path)
                    logger.info(f"Copied profile for {speaker}: {output_path} ({duration:.2f}s)")
                except Exception as e2:
                    logger.error(f"Failed to copy file for {speaker}: {str(e2)}")
        
        return len(speaker_files)
        
    except Exception as e:
        logger.error(f"Error extracting speaker profiles: {str(e)}")
        return 0

def main():
    parser = argparse.ArgumentParser(description="Filter manifest by speaker duration and extract speaker profiles")
    parser.add_argument("--speaker-info", type=str, required=True, help="Path to speaker info text file")
    parser.add_argument("--input-manifest", type=str, required=True, help="Path to input manifest file")
    parser.add_argument("--output-manifest", type=str, required=True, help="Path to output filtered manifest file")
    parser.add_argument("--profile-dir", type=str, default="speaker_profile", help="Directory to save speaker profile audio")
    parser.add_argument("--min-duration", type=float, default=7.0, help="Minimum duration for speaker profile audio (seconds)")
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(os.path.dirname(args.output_manifest), exist_ok=True)
    os.makedirs(args.profile_dir, exist_ok=True)
    
    # Parse speaker info
    logger.info(f"Parsing speaker info from {args.speaker_info}")
    selected_speakers = parse_speaker_info(args.speaker_info)
    
    if not selected_speakers:
        logger.error("No eligible speakers found. Exiting.")
        return
    
    # Filter manifest
    logger.info(f"Filtering manifest {args.input_manifest} -> {args.output_manifest}")
    filtered_count = filter_manifest(args.input_manifest, args.output_manifest, selected_speakers)
    
    if filtered_count == 0:
        logger.warning("No entries were filtered. Check speaker names in manifest.")
    
    # Extract speaker profiles
    logger.info(f"Extracting speaker profiles to {args.profile_dir}")
    profile_count = extract_speaker_profiles(
        args.input_manifest, 
        args.profile_dir, 
        selected_speakers, 
        args.min_duration
    )
    
    logger.info(f"Process complete. Filtered {filtered_count} entries and extracted {profile_count} speaker profiles.")

if __name__ == "__main__":
    main()