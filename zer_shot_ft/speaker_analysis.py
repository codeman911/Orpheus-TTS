import json
import os
import argparse
from collections import defaultdict
from tqdm import tqdm
import logging
import wave
import contextlib
import soundfile as sf

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def get_audio_duration(audio_path):
    """Get the duration of an audio file in seconds"""
    try:
        # Handle WAV files with wave module
        if audio_path.lower().endswith('.wav'):
            try:
                with contextlib.closing(wave.open(audio_path, 'r')) as f:
                    frames = f.getnframes()
                    rate = f.getframerate()
                    duration = frames / float(rate)
                    return duration
            except Exception as wave_error:
                # If wave module fails, try soundfile as fallback
                logger.debug(f"Wave module failed for {audio_path}: {str(wave_error)}. Trying soundfile...")
                try:
                    info = sf.info(audio_path)
                    return info.duration
                except Exception as sf_error:
                    # If both fail, try librosa as a last resort
                    logger.debug(f"Soundfile also failed: {str(sf_error)}. Trying librosa...")
                    try:
                        import librosa
                        duration = librosa.get_duration(path=audio_path)
                        return duration
                    except Exception as librosa_error:
                        logger.warning(f"All methods failed to get duration for {audio_path}. Last error: {str(librosa_error)}")
                        # As a last resort, estimate from file size (rough approximation)
                        try:
                            # For 44.1kHz, 16-bit stereo WAV, ~176KB per second
                            file_size = os.path.getsize(audio_path)
                            estimated_duration = file_size / (44100 * 2 * 2)  # Sample rate * bytes per sample * channels
                            logger.info(f"Estimated duration from file size: {estimated_duration:.2f}s for {audio_path}")
                            return estimated_duration if estimated_duration > 0 else 0
                        except Exception:
                            return 0
        # Handle other audio formats with soundfile
        else:
            try:
                info = sf.info(audio_path)
                return info.duration
            except Exception as e:
                logger.debug(f"Soundfile failed for {audio_path}: {str(e)}. Trying librosa...")
                try:
                    import librosa
                    duration = librosa.get_duration(path=audio_path)
                    return duration
                except Exception as librosa_error:
                    logger.warning(f"All methods failed to get duration for {audio_path}. Last error: {str(librosa_error)}")
                    return 0
    except Exception as e:
        logger.debug(f"Error getting duration for {audio_path}: {str(e)}")
        return 0

def read_manifest_file(manifest_path):
    """Read manifest file and extract speaker information and durations"""
    speaker_data = defaultdict(float)
    total_files = 0
    processed_files = 0
    skipped_files = 0
    
    # Track reasons for skipping
    skip_reasons = defaultdict(int)
    
    # Get the directory of the manifest file to resolve relative paths
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    
    logger.info(f"Reading manifest file: {manifest_path}")
    
    try:
        with open(manifest_path, 'r') as f:
            for line in tqdm(f, desc="Processing manifest entries"):
                total_files += 1
                
                if not line.strip():
                    continue
                    
                try:
                    item = json.loads(line)
                    
                    # Get speaker information - use speaker, voice_name, or a default value
                    speaker = item.get("speaker", item.get("voice_name", "unknown_speaker"))
                    
                    # Get audio file path
                    audio_path = item.get("audio_filepath")
                    if not audio_path:
                        skip_reasons["missing_path"] += 1
                        skipped_files += 1
                        continue
                    
                    # Try to resolve the path if it's relative
                    if not os.path.isabs(audio_path):
                        # First try relative to manifest directory
                        full_path = os.path.join(manifest_dir, audio_path)
                        if not os.path.exists(full_path):
                            # Try with parent directory (for ../datasets structure)
                            parent_dir = os.path.dirname(manifest_dir)
                            full_path = os.path.join(parent_dir, audio_path)
                            
                            # If still not found, try with absolute path from root
                            if not os.path.exists(full_path):
                                # Try to find the file by its basename in common audio directories
                                basename = os.path.basename(audio_path)
                                for root_dir in ["/vast/audio/data", "/vast/audio/experiment"]:
                                    for root, _, files in os.walk(root_dir):
                                        if basename in files:
                                            full_path = os.path.join(root, basename)
                                            logger.debug(f"Found file at: {full_path}")
                                            break
                                    if os.path.exists(full_path):
                                        break
                        
                        audio_path = full_path
                    
                    # Check if audio file exists
                    if not os.path.exists(audio_path):
                        skip_reasons["file_not_found"] += 1
                        skipped_files += 1
                        # Log a sample of missing files
                        if skip_reasons["file_not_found"] <= 5:
                            logger.warning(f"File not found: {audio_path}")
                        continue
                    
                    # Calculate duration directly from the audio file
                    duration = get_audio_duration(audio_path)
                    if duration <= 0 and USE_FILESIZE_FALLBACK:
                        # Fallback to file size estimation
                        try:
                            file_size = os.path.getsize(audio_path)
                            # For 44.1kHz, 16-bit stereo WAV, ~176KB per second
                            estimated_duration = file_size / (44100 * 2 * 2)  # Sample rate * bytes per sample * channels
                            logger.debug(f"Using file size estimation for {audio_path}: {estimated_duration:.2f}s")
                            duration = estimated_duration
                        except Exception as e:
                            logger.debug(f"File size estimation failed: {str(e)}")
                    
                    if duration <= 0:
                        skip_reasons["invalid_duration"] += 1
                        skipped_files += 1
                        continue
                    
                    # Add duration to speaker's total
                    speaker_data[speaker] += duration
                    processed_files += 1
                    
                except json.JSONDecodeError:
                    skip_reasons["json_error"] += 1
                    skipped_files += 1
                except Exception as e:
                    skip_reasons["other_error"] += 1
                    if skip_reasons["other_error"] <= 5:
                        logger.warning(f"Error processing entry: {str(e)}")
                    skipped_files += 1
    
    except Exception as e:
        logger.error(f"Error reading manifest file: {str(e)}")
        raise
    
    # Log detailed skip reasons
    logger.info(f"Processed {processed_files} files, skipped {skipped_files} files")
    for reason, count in skip_reasons.items():
        logger.info(f"  - Skipped due to {reason}: {count}")
    
    # If all files were skipped, print a sample of the manifest for debugging
    if processed_files == 0 and total_files > 0:
        logger.warning("All files were skipped. Printing sample of manifest for debugging:")
        try:
            with open(manifest_path, 'r') as f:
                sample = [next(f) for _ in range(min(3, total_files))]
                for i, line in enumerate(sample):
                    logger.warning(f"Sample {i+1}: {line.strip()}")
        except Exception:
            pass
    
    return speaker_data

# Define USE_FILESIZE_FALLBACK as a global variable
USE_FILESIZE_FALLBACK = False

def analyze_speakers(manifest_paths, output_file, use_filesize_fallback=False):
    """Analyze speakers from multiple manifest files and save results"""
    global USE_FILESIZE_FALLBACK
    USE_FILESIZE_FALLBACK = use_filesize_fallback
    
    if use_filesize_fallback:
        logger.info("File size fallback enabled for duration estimation")
    
    all_speaker_data = defaultdict(float)
    
    # Process each manifest file
    for manifest_path in manifest_paths:
        if not os.path.exists(manifest_path):
            logger.warning(f"Manifest file not found: {manifest_path}")
            continue
            
        speaker_data = read_manifest_file(manifest_path)
        
        # Merge with overall data
        for speaker, duration in speaker_data.items():
            all_speaker_data[speaker] += duration
    
    # Convert seconds to hours and sort by duration (descending)
    speaker_hours = {speaker: duration / 3600 for speaker, duration in all_speaker_data.items()}
    sorted_speakers = sorted(speaker_hours.items(), key=lambda x: x[1], reverse=True)
    
    # Save results to file
    with open(output_file, 'w') as f:
        f.write("Speaker Analysis Results (sorted by duration)\n")
        f.write("===========================================\n\n")
        f.write(f"Total speakers found: {len(sorted_speakers)}\n\n")
        f.write("Speaker Name                                  | Duration (hours)\n")
        f.write("-----------------------------------------------+----------------\n")
        
        for speaker, hours in sorted_speakers:
            # Format with fixed width for better readability
            f.write(f"{speaker:<48} | {hours:>14.2f}\n")
    
    logger.info(f"Analysis complete. Results saved to {output_file}")
    
    # Print summary
    total_hours = sum(speaker_hours.values())
    logger.info(f"Total speakers: {len(sorted_speakers)}")
    logger.info(f"Total audio duration: {total_hours:.2f} hours")
    
    # Print top 10 speakers
    logger.info("Top 10 speakers by duration:")
    for i, (speaker, hours) in enumerate(sorted_speakers[:10], 1):
        logger.info(f"{i}. {speaker}: {hours:.2f} hours")

# In the main function, add this argument
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze speaker information from manifest files")
    parser.add_argument("--manifests", type=str, nargs='+', required=True, 
                        help="Paths to manifest files (space-separated)")
    parser.add_argument("--output", type=str, default="speaker_info.txt", 
                        help="Output file path for speaker analysis")
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="Number of files to process in each batch")
    parser.add_argument("--use-filesize-fallback", action="store_true",
                        help="Use file size to estimate duration if audio libraries fail")
    
    args = parser.parse_args()
    
    # Pass the fallback flag to the analyze_speakers function
    analyze_speakers(args.manifests, args.output, use_filesize_fallback=args.use_filesize_fallback)