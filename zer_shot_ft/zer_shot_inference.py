import re
import os
import sys
import wave
import time
import torch
import argparse
import numpy as np
import torchaudio
import torchaudio.transforms as T
import librosa
import logging
from orpheus_tts import OrpheusModel
from transformers import AutoTokenizer
from snac import SNAC

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class ZeroShotOrpheusModel(OrpheusModel):
    def __init__(self, model_name, dtype=torch.bfloat16):
        """Initialize the zero-shot voice cloning model"""
        super().__init__(model_name, dtype)
        
        # Override validation to be more permissive
        self._original_validate = self.validate_voice
        self.validate_voice = lambda voice: None
        
        # Load the tokenizer directly
        self.direct_tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add custom voices to the available voices list - match training data
        self.available_voices.extend([
            "Elise", "8OqhOrMyhp0_SPEAKER_01_0", "1bBRyX90RAw_SPEAKER_05_0", 
            "spk_ar_rf_1548", "spk_sp_ar_7", "spk_sp_ar_39", "Emirati_female_1",
            "Emirati_male_1", "Emirati_male_2", "Emirati_male_3", "Emirati_male_6",
            "Emirati_1", "spk_sp_en_2", "Nouran", "Nabeel", "Emirati_10"
        ])
        logger.info(f"Available voices: {self.available_voices}")
        
        # Load SNAC model for audio encoding/decoding
        logger.info("Loading SNAC model...")
        self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
        if torch.cuda.is_available():
            self.snac_model = self.snac_model.to("cuda:0")
        else:
            self.snac_model = self.snac_model.to("cpu")
        logger.info("SNAC model loaded successfully")
        
        # Special token IDs - exactly match those in zero_shot_datasets.py
        self.start_of_human = 128259
        self.end_of_text = 128009
        self.end_of_human = 128260
        self.start_of_ai = 128261
        self.start_of_speech = 128257
        self.end_of_speech = 128258
        self.end_of_ai = 128262
        self.audio_tokens_start = 128266
    
    def tokenise_audio(self, waveform, sample_rate=24000):
        """Convert audio waveform to SNAC tokens - exactly match zero_shot_datasets.py"""
        try:
            # Convert to numpy if it's a torch tensor
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.squeeze().numpy()  # Ensure we have a 1D array
            
            # Ensure we have a numpy array
            waveform = np.asarray(waveform, dtype=np.float32)
            
            # Normalize audio - exactly as in zero_shot_datasets.py
            waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
            
            # Convert to torch tensor in the exact format needed
            waveform = torch.from_numpy(waveform).unsqueeze(0)  # [1, T]
            waveform = waveform.to(dtype=torch.float32)
            
            # Resample if needed - match training pipeline
            if sample_rate != 24000:
                logger.info(f"Resampling from {sample_rate} to 24000 Hz")
                resample_transform = T.Resample(orig_freq=sample_rate, new_freq=24000)
                waveform = resample_transform(waveform)
            
            # Ensure correct shape for SNAC model [B, 1, T]
            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(1)  # [B, 1, T]
            
            # Move to appropriate device
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            waveform = waveform.to(device)
            
            # Get SNAC codes - exactly as in zero_shot_datasets.py
            with torch.inference_mode():
                codes = self.snac_model.encode(waveform)
            
            # Format codes exactly as in zero_shot_datasets.py
            all_codes = []
            for i in range(codes[0].shape[1]):
                # Only add if we have complete data for this frame
                if (4*i)+3 < codes[2][0].shape[0] and (2*i)+1 < codes[1][0].shape[0]:
                    all_codes.append(int(codes[0][0][i].item()) + self.audio_tokens_start)
                    all_codes.append(int(codes[1][0][2*i].item()) + self.audio_tokens_start + 4096)
                    all_codes.append(int(codes[2][0][4*i].item()) + self.audio_tokens_start + (2*4096))
                    all_codes.append(int(codes[2][0][(4*i)+1].item()) + self.audio_tokens_start + (3*4096))
                    all_codes.append(int(codes[1][0][(2*i)+1].item()) + self.audio_tokens_start + (4*4096))
                    all_codes.append(int(codes[2][0][(4*i)+2].item()) + self.audio_tokens_start + (5*4096))
                    all_codes.append(int(codes[2][0][(4*i)+3].item()) + self.audio_tokens_start + (6*4096))
            
            logger.info(f"Generated {len(all_codes)} audio tokens")
            return all_codes
        except Exception as e:
            logger.error(f"Error in tokenise_audio: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def remove_duplicate_frames(self, codes_list):
        """Remove duplicate frames from codes_list - match training pipeline"""
        if codes_list is None or len(codes_list) == 0:
            return None
            
        if len(codes_list) % 7 != 0:
            # Truncate to nearest multiple of 7
            original_length = len(codes_list)
            codes_list = codes_list[:-(len(codes_list) % 7)]
            logger.info(f"Truncated audio tokens from {original_length} to {len(codes_list)} to ensure complete frames")
        
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
        
        logger.info(f"After removing duplicates: {len(result)} audio tokens")
        return result
    
    def decode_speech_tokens(self, token_ids):
        """Decode speech tokens to audio bytes - using Trelis approach"""
        try:
            # Convert to numpy array if it's a tensor
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.cpu().numpy()
            
            # Filter out non-audio tokens
            audio_tokens = []
            for token in token_ids:
                if token >= self.audio_tokens_start and token < self.audio_tokens_start + (7*4096):
                    audio_tokens.append(token)
            
            if not audio_tokens:
                logger.warning("No audio tokens found in generated output")
                return b''
            
            # Group tokens into frames of 7
            frames = []
            for i in range(0, len(audio_tokens) - (len(audio_tokens) % 7), 7):
                frame = audio_tokens[i:i+7]
                if len(frame) == 7:  # Only add complete frames
                    frames.append(frame)
            
            if not frames:
                logger.warning("No complete audio frames found")
                return b''
            
            # Prepare tokens for SNAC decoding
            codes = [[], [], []]
            
            for frame in frames:
                codes[0].append(frame[0] - self.audio_tokens_start)
                codes[1].append(frame[1] - self.audio_tokens_start - 4096)
                codes[1].append(frame[4] - self.audio_tokens_start - (4*4096))
                codes[2].append(frame[2] - self.audio_tokens_start - (2*4096))
                codes[2].append(frame[3] - self.audio_tokens_start - (3*4096))
                codes[2].append(frame[5] - self.audio_tokens_start - (5*4096))
                codes[2].append(frame[6] - self.audio_tokens_start - (6*4096))
            
            # Convert to tensors
            codes = [torch.tensor(c, dtype=torch.long).unsqueeze(0) for c in codes]
            
            # Decode audio
            with torch.inference_mode():
                waveform = self.snac_model.decode(codes)
            
            # Convert to 16-bit PCM
            waveform = waveform.cpu().numpy().squeeze()
            waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)  # Normalize
            waveform = (waveform * 32767).astype(np.int16)
            
            # Convert to bytes
            return waveform.tobytes()
        except Exception as e:
            logger.error(f"Error decoding speech tokens: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return b''
    
    def format_cloning_prompt(self, reference_text, reference_audio_tokens, new_text):
        """Format prompt for zero-shot voice cloning - exactly match zero_shot_datasets.py"""
        # Create token IDs from config - using the same format as in zero_shot_datasets.py
        start_of_human = self.start_of_human
        end_of_human = self.end_of_human
        start_of_ai = self.start_of_ai
        start_of_speech = self.start_of_speech
        end_of_speech = self.end_of_speech
        end_of_ai = self.end_of_ai
        end_of_text = self.end_of_text
        
        # Encode reference text
        reference_text_ids = self.direct_tokenizer.encode(reference_text, add_special_tokens=False)
        
        # Encode target text
        target_text_ids = self.direct_tokenizer.encode(new_text, add_special_tokens=False)
        
        # Create zero-shot voice cloning format exactly like in zero_shot_datasets.py
        # Format: start_of_human, text, end_of_text, end_of_human, start_of_ai, start_of_speech, 
        #         speech, end_of_speech, end_of_ai, start_of_human, text, end_of_text, end_of_human
        input_ids = (
            [start_of_human] 
            + reference_text_ids 
            + [end_of_text, end_of_human]
            + [start_of_ai] 
            + [start_of_speech] 
            + reference_audio_tokens 
            + [end_of_speech] 
            + [end_of_ai]
            + [start_of_human] 
            + target_text_ids 
            + [end_of_text, end_of_human]
        )
        
        # Convert to tensor for the model
        input_ids_tensor = torch.tensor([input_ids], dtype=torch.long)
        
        # Log token structure for debugging
        logger.info(f"=== Zero-Shot Prompt Structure ===")
        logger.info(f"Reference Text: {reference_text}")
        logger.info(f"Target Text: {new_text}")
        
        # Log token counts and structure
        ref_audio_len = len(reference_audio_tokens)
        
        logger.info(f"Token Structure:")
        logger.info(f"  - start_of_human: {start_of_human}")
        logger.info(f"  - reference_text_ids: {len(reference_text_ids)} tokens")
        logger.info(f"  - end_of_text: {end_of_text}, end_of_human: {end_of_human}")
        logger.info(f"  - start_of_ai: {start_of_ai}, start_of_speech: {start_of_speech}")
        logger.info(f"  - reference_audio: {ref_audio_len} tokens")
        logger.info(f"  - end_of_speech: {end_of_speech}, end_of_ai: {end_of_ai}")
        logger.info(f"  - start_of_human: {start_of_human}")
        logger.info(f"  - target_text_ids: {len(target_text_ids)} tokens")
        logger.info(f"  - end_of_text: {end_of_text}, end_of_human: {end_of_human}")
        
        # Log total length
        logger.info(f"Total Input IDs Length: {len(input_ids)}")
        
        # Convert to string for the model
        prompt_string = self.direct_tokenizer.decode(input_ids_tensor[0], skip_special_tokens=False)
        
        # Remove any <s> tags that might have been added by the tokenizer
        prompt_string = prompt_string.replace("<s>", "").replace("</s>", "")
        
        return prompt_string, input_ids_tensor
    
    def format_prompt_with_voice(self, prompt, voice):
        """Format prompt with a specific voice (for standard TTS)"""
        # Don't add voice prefix - model wasn't trained this way
        # Just tokenize the text directly
        prompt_tokens = self.direct_tokenizer(prompt, return_tensors="pt").input_ids
        
        # Add special tokens exactly as in zero_shot_datasets.py
        start_token = torch.tensor([[self.start_of_human]], dtype=torch.int64)
        end_tokens = torch.tensor([[self.end_of_text, self.end_of_human, self.start_of_ai, self.start_of_speech]], dtype=torch.int64)
        
        # Combine tokens
        all_input_ids = torch.cat([start_token, prompt_tokens, end_tokens], dim=1)
        
        # Convert to string for the model - IMPORTANT: skip_special_tokens must be False
        # but we need to ensure no <s> tag is added
        prompt_string = self.direct_tokenizer.decode(all_input_ids[0], skip_special_tokens=False)
        
        # Remove any <s> tags that might have been added by the tokenizer
        prompt_string = prompt_string.replace("<s>", "").replace("</s>", "")
        
        # Log for debugging
        logger.info(f"Using voice: {voice}")
        logger.info(f"Prompt format (first 50 chars): {prompt_string[:50]}...")
        
        return prompt_string
    
    def generate_with_voice(self, prompt, voice, output_path=None, **kwargs):
        """Generate speech using a specific voice"""
        logger.info(f"Generating speech with voice: {voice}")
        
        # Format the prompt with voice
        formatted_prompt = self.format_prompt_with_voice(prompt, voice)
        
        # Set default generation parameters if not provided
        if 'repetition_penalty' not in kwargs:
            kwargs['repetition_penalty'] = 1.1
        if 'stop_token_ids' not in kwargs:
            kwargs['stop_token_ids'] = [self.end_of_speech]
        if 'max_tokens' not in kwargs:
            kwargs['max_tokens'] = 8192
        if 'temperature' not in kwargs:
            kwargs['temperature'] = 0.4
        if 'top_p' not in kwargs:
            kwargs['top_p'] = 0.9
        
        # Generate speech
        start_time = time.monotonic()
        syn_tokens = self.generate_speech(
            prompt=formatted_prompt,
            **kwargs
        )
        
        # Save to file if output path is provided
        if output_path:
            self._save_audio(syn_tokens, output_path, start_time)
        
        return syn_tokens
    
    def clone_voice(self, reference_audio_path, reference_text, new_text, output_path=None, **kwargs):
        """Generate speech in the voice of the reference audio - exactly match zero_shot_datasets.py"""
        logger.info(f"Loading reference audio from {reference_audio_path}")
        
        try:
            # Load audio using torchaudio to match zero_shot_datasets.py
            waveform, sample_rate = torchaudio.load(reference_audio_path)
            
            # Process audio to match training pipeline
            if waveform.shape[0] > 1:  # Convert stereo to mono
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Trim silence from the beginning and end
            waveform = waveform.squeeze()  # Remove batch dimension for processing
            
            # Normalize audio before tokenization
            waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
            
            # Get SNAC tokens from reference audio
            logger.info("Tokenizing reference audio...")
            audio_tokens = self.tokenise_audio(waveform, sample_rate)
            if audio_tokens is None or len(audio_tokens) == 0:
                raise ValueError("Failed to extract audio tokens from reference audio")
            
            # Remove duplicate frames for more efficient processing
            audio_tokens = self.remove_duplicate_frames(audio_tokens)
            if audio_tokens is None or len(audio_tokens) == 0:
                raise ValueError("Failed to process audio tokens after removing duplicates")
            
            # Limit token length to avoid context window issues
            max_audio_tokens = 2048  # Reasonable limit for audio tokens
            if len(audio_tokens) > max_audio_tokens:
                logger.warning(f"Truncating audio tokens from {len(audio_tokens)} to {max_audio_tokens}")
                audio_tokens = audio_tokens[:max_audio_tokens - (max_audio_tokens % 7)]  # Ensure multiple of 7
            
            logger.info(f"Using {len(audio_tokens)} audio tokens from reference audio")
            
            # Format the prompt for zero-shot cloning
            logger.info("Formatting prompt for voice cloning...")
            formatted_prompt, input_ids = self.format_cloning_prompt(reference_text, audio_tokens, new_text)
            
            # Log the first few tokens for debugging
            logger.info(f"First 10 tokens: {input_ids[0][:10].tolist()}")
            logger.info(f"Last 10 tokens: {input_ids[0][-10:].tolist()}")
            
            # Set default generation parameters if not provided
            if 'repetition_penalty' not in kwargs:
                kwargs['repetition_penalty'] = 1.3
            if 'stop_token_ids' not in kwargs:
                kwargs['stop_token_ids'] = [self.end_of_speech]
            if 'max_tokens' not in kwargs:
                kwargs['max_tokens'] = 8192
            if 'temperature' not in kwargs:
                kwargs['temperature'] = 0.7
            if 'top_p' not in kwargs:
                kwargs['top_p'] = 0.9
            
            # Generate speech with the cloned voice
            logger.info("Generating speech with cloned voice...")
            start_time = time.monotonic()
            syn_tokens = self.generate_speech(
                prompt=formatted_prompt,
                **kwargs
            )
            
            # Save to file if output path is provided
            if output_path:
                self._save_audio(syn_tokens, output_path, start_time)
            
            return syn_tokens
        except Exception as e:
            logger.error(f"Error in clone_voice: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def generate_speech(self, **kwargs):
        """Generate speech from prompt - using approach from test_inference.py"""
        try:
            # Extract prompt from kwargs
            prompt = kwargs.pop('prompt', None)
            if prompt is None:
                raise ValueError("No prompt provided for speech generation")
            
            # Set up generation parameters
            temperature = kwargs.pop('temperature', 0.7)
            top_p = kwargs.pop('top_p', 0.9)
            repetition_penalty = kwargs.pop('repetition_penalty', 1.3)
            max_tokens = kwargs.pop('max_tokens', 8192)
            stop_token_ids = kwargs.pop('stop_token_ids', [self.end_of_speech])
            
            # Log generation parameters
            logger.info(f"Generating speech with parameters: temp={temperature}, top_p={top_p}, rep_penalty={repetition_penalty}")
            
            # Get device directly from the model
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            
            # Tokenize the prompt and move to the correct device
            input_ids = self.direct_tokenizer.encode(prompt, return_tensors="pt").to(device)
            
            # Generate speech tokens
            with torch.inference_mode():
                outputs = self.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    eos_token_id=stop_token_ids,
                    pad_token_id=self.direct_tokenizer.pad_token_id
                )
            
            # Extract generated tokens (excluding prompt)
            generated_ids = outputs[0][input_ids.shape[1]:]
            
            # Find where the speech ends (at end_of_speech token)
            try:
                end_idx = (generated_ids == self.end_of_speech).nonzero()[0].item()
                generated_ids = generated_ids[:end_idx]
            except (IndexError, ValueError):
                # No end_of_speech token found, use all tokens
                pass
            
            # Convert to audio
            audio_bytes = self.decode_speech_tokens(generated_ids)
            
            # Log success
            logger.info(f"Successfully generated speech")
            
            # Return audio bytes as an iterable (to match parent class interface)
            return [audio_bytes]
        except Exception as e:
            logger.error(f"Error in generate_speech: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _save_audio(self, syn_tokens, output_path, start_time=None):
        """Save generated audio to a WAV file and log statistics"""
        try:
            if syn_tokens is None:
                logger.error("No audio tokens to save")
                return
            
            # Check if we have any audio data
            audio_data = b''.join(list(syn_tokens))
            if not audio_data:
                logger.error("No audio data generated")
                return
            
            with wave.open(output_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(audio_data)
                
                # Calculate duration safely
                duration = len(audio_data) / (24000 * 2)  # bytes / (sample_rate * bytes_per_sample)
                
                if start_time and duration > 0:
                    end_time = time.monotonic()
                    logger.info(f"Generated {duration:.2f} seconds of audio in {end_time - start_time:.2f} seconds")
                    logger.info(f"Real-time factor: {(end_time - start_time) / duration:.2f}x")
                
                logger.info(f"Audio saved to {output_path}")
                
                # Verify the file was created successfully
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    logger.info(f"Successfully saved audio file ({os.path.getsize(output_path)} bytes)")
                else:
                    logger.error(f"Failed to save audio file or file is empty")
        except Exception as e:
            logger.error(f"Error saving audio: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())


def main():
    parser = argparse.ArgumentParser(description="Zero-shot voice cloning with Orpheus TTS")
    parser.add_argument("--model", type=str, 
                        default="/vast/audio/experiment/Orpheus-TTS/zer_shot_ft/checkpoints_zr/checkpoint-12000", 
                        help="Path to the model checkpoint")
    
    # Common parameters
    parser.add_argument("--text", type=str,
                        default="Man, the way social media has, um, completely changed how we interact is just wild, right? Like, we're all connected 24/7 but somehow people feel more alone than ever. And don't even get me started on how it's messing with kids' self-esteem and mental health and whatnot.", 
                        help="Text to synthesize")
    parser.add_argument("--output", type=str, 
                        default="../output_zs.wav", 
                        help="Output audio file path")
    parser.add_argument("--temperature", type=float, 
                        default=0.4, 
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, 
                        default=0.95, 
                        help="Top-p sampling parameter")
    parser.add_argument("--repetition_penalty", type=float,
                        default=1.3,
                        help="Repetition penalty for generation")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode with additional logging")
    
    # Mode selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--voice", type=str,
                      help="Use a predefined voice (e.g., 'Emirati_female_1')")
    group.add_argument("--reference", type=str,
                      help="Path to reference audio file for zero-shot cloning")
    
    # Zero-shot cloning parameters
    parser.add_argument("--reference_text", type=str,
                        default="""ويمكن موضوع الملوحة في الماء. بس أنت يعني تعطينا Tips في صناعة المحتوى الزراعي عن كيف أننا نستعمل هذه التقنيات وكيف نستعمل، كيف نتعامل مع الفشل في الزراعة. شو ممكن تخبرنا عن صناعة المحتوى؟""",  
                        help="Transcript of the reference audio (required for --reference)")
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    try:
        # Initialize model
        model = ZeroShotOrpheusModel(model_name=args.model)
        
        # Generate speech based on selected mode
        if args.voice:
            # Standard TTS with predefined voice
            model.generate_with_voice(
                prompt=args.text,
                voice=args.voice,
                output_path=args.output,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty
            )
        elif args.reference:
            # Zero-shot voice cloning
            if not args.reference_text:
                parser.error("--reference_text is required when using --reference")
                
            model.clone_voice(
                reference_audio_path=args.reference,
                reference_text=args.reference_text,
                new_text=args.text,
                output_path=args.output,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty
            )
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()