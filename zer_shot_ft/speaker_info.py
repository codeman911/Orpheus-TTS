import torch
from orpheus_tts import OrpheusModel
import os
import re
from transformers import AutoTokenizer

class SpeakerInfoExtractor(OrpheusModel):
    def __init__(self, model_name, dtype=torch.bfloat16):
        super().__init__(model_name, dtype)
        # Disable validation to avoid errors
        self._original_validate = self.validate_voice
        self.validate_voice = lambda voice: None
        
    def extract_speakers_from_vocab(self):
        """Extract potential speaker names from the tokenizer vocabulary"""
        # Get all tokens from the tokenizer
        vocab = self.tokeniser.get_vocab()
        
        # Look for tokens that might be speaker names (simple heuristic)
        potential_speakers = []
        for token, _ in vocab.items():
            # Skip special tokens and very short tokens
            if token.startswith('<') or len(token) < 3:
                continue
            
            # Look for tokens that end with a colon (typical speaker format)
            if token.endswith(':'):
                speaker = token.rstrip(':').strip()
                if speaker:
                    potential_speakers.append(speaker)
            
            # Also look for tokens that might be speaker IDs
            if re.match(r'^[A-Za-z0-9_]+$', token) and len(token) > 5:
                if '_SPEAKER_' in token or any(c.isupper() for c in token):
                    potential_speakers.append(token)
        
        return sorted(set(potential_speakers))

def main():
    # Path to your model
    model_path = "/vast/audio/experiment/Orpheus-TTS/speaker_finetune/checkpoints/checkpoint-60000"
    
    print(f"Loading model from {model_path}...")
    extractor = SpeakerInfoExtractor(model_name=model_path)
    
    # Get default available voices
    default_voices = extractor.available_voices
    print(f"\nDefault available voices: {default_voices}")
    
    # Extract potential speakers from vocabulary
    print("\nExtracting potential speaker names from vocabulary...")
    potential_speakers = extractor.extract_speakers_from_vocab()
    print(f"Found {len(potential_speakers)} potential speaker names")
    
    # Save to file
    output_file = "available_speakers.txt"
    with open(output_file, "w") as f:
        f.write("# Default Available Voices\n")
        for voice in default_voices:
            f.write(f"{voice}\n")
        
        f.write("\n# Potential Additional Speakers (extracted from vocabulary)\n")
        for speaker in potential_speakers:
            f.write(f"{speaker}\n")
    
    print(f"\nSpeaker information saved to {output_file}")
    
    # Print some example usage
    print("\nExample usage in inference_test.py:")
    print("----------------------------------------")
    print("# Add these speakers to your CustomOrpheusModel:")
    print("self.available_voices.extend([")
    for speaker in potential_speakers[:5]:  # Show first 5 as examples
        print(f'    "{speaker}",')
    print("    # ... add more as needed")
    print("])")

if __name__ == "__main__":
    main()