import re
import os
import wave
import time
import torch
import argparse
import numpy as np
import torchaudio
import torchaudio.transforms as T
from orpheus_tts import OrpheusModel
from transformers import AutoTokenizer
from snac import SNAC
import logging
import librosa  # Add librosa import

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class ZeroShotCloningModel(OrpheusModel):
    def __init__(self, model_name, dtype=torch.bfloat16):
        """Initialize the zero-shot voice cloning model"""
        super().__init__(model_name, dtype)
        
        # Override validation to be more permissive
        self._original_validate = self.validate_voice
        self.validate_voice = lambda voice: None
        
        # Load the tokenizer directly
        self.direct_tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add custom voices to the available voices list
        self.available_voices.extend([
            "Elise", "8OqhOrMyhp0_SPEAKER_01_0", "1bBRyX90RAw_SPEAKER_05_0", 
            "spk_ar_rf_1548", "spk_sp_ar_7","spk_sp_ar_39"
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
        
        # Special token IDs
        self.start_of_human = 128259
        self.end_of_text = 128009
        self.end_of_human = 128260
        self.start_of_ai = 128261
        self.start_of_speech = 128257
        self.end_of_speech = 128258
        self.end_of_ai = 128262
        self.audio_tokens_start = 128266
        
    # Add format_prompt_with_voice method from inference_test.py
    def format_prompt_with_voice(self, prompt, voice):
        """Format prompt with voice prefix for speaker-based generation"""
        # Format with voice prefix
        adapted_prompt = f"{voice}: {prompt}"
        
        # Encode text with speaker information
        prompt_tokens = self.direct_tokenizer(adapted_prompt, return_tensors="pt")
        
        # Add special tokens
        start_token = torch.tensor([[self.start_of_human]], dtype=torch.int64)
        end_tokens = torch.tensor([[self.end_of_text, self.end_of_human, self.start_of_ai, self.start_of_speech]], dtype=torch.int64)
        
        # Combine tokens
        all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
        
        # Convert to string for the model
        prompt_string = self.direct_tokenizer.decode(all_input_ids[0])
        return prompt_string
    
    # Add generate_with_voice method
    def generate_with_voice(self, prompt, voice, output_path=None, **kwargs):
        """Generate speech using a specific voice"""
        logger.info(f"Generating speech with voice: {voice}")
        
        # Format the prompt with voice
        formatted_prompt = self.format_prompt_with_voice(prompt, voice)
        
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
        
        # Generate speech
        start_time = time.monotonic()
        syn_tokens = self.generate_speech(
            prompt=formatted_prompt,
            **kwargs
        )
        
        # Save audio if output path is provided
        if output_path:
            with wave.open(output_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                
                total_frames = 0
                for audio_chunk in syn_tokens:
                    frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
                    total_frames += frame_count
                    wf.writeframes(audio_chunk)
                
                duration = total_frames / wf.getframerate()
                end_time = time.monotonic()
                logger.info(f"Generated {duration:.2f} seconds of audio in {end_time - start_time:.2f} seconds")
                logger.info(f"Real-time factor: {(end_time - start_time) / duration:.2f}x")
                logger.info(f"Audio saved to {output_path}")
        
        return syn_tokens
    
    def tokenise_audio(self, waveform, sample_rate=24000):
        """Convert audio waveform to SNAC tokens using Trelis approach"""
        try:
            # Convert to numpy if it's a torch tensor
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.numpy()
            
            # Ensure we have a numpy array
            waveform = np.asarray(waveform, dtype=np.float32)
            
            # Convert to torch tensor in the exact format Trelis uses
            waveform = torch.from_numpy(waveform).unsqueeze(0)
            waveform = waveform.to(dtype=torch.float32)
            
            # Resample if needed
            if sample_rate != 24000:
                resample_transform = T.Resample(orig_freq=sample_rate, new_freq=24000)
                waveform = resample_transform(waveform)
            
            # Add batch dimension exactly as in Trelis code
            waveform = waveform.unsqueeze(0)
            
            # Move to appropriate device
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            waveform = waveform.to(device)
            
            with torch.inference_mode():
                codes = self.snac_model.encode(waveform)
            
            # Follow Trelis approach for token encoding exactly
            all_codes = []
            for i in range(codes[0].shape[1]):
                all_codes.append(codes[0][0][i].item() + self.audio_tokens_start)
                all_codes.append(codes[1][0][2*i].item() + self.audio_tokens_start + 4096)
                all_codes.append(codes[2][0][4*i].item() + self.audio_tokens_start + (2*4096))
                all_codes.append(codes[2][0][(4*i)+1].item() + self.audio_tokens_start + (3*4096))
                all_codes.append(codes[1][0][(2*i)+1].item() + self.audio_tokens_start + (4*4096))
                all_codes.append(codes[2][0][(4*i)+2].item() + self.audio_tokens_start + (5*4096))
                all_codes.append(codes[2][0][(4*i)+3].item() + self.audio_tokens_start + (6*4096))
        
            return all_codes
        except Exception as e:
            logger.error(f"Error in tokenise_audio: {str(e)}")
            return None
    
    def remove_duplicate_frames(self, codes_list):
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
    
    def format_cloning_prompt(self, reference_text, reference_audio_tokens, new_text):
        """Format prompt for zero-shot voice cloning using Trelis approach"""
        # Create special tokens
        start_token = torch.tensor([[self.start_of_human]], dtype=torch.int64)
        mid_tokens = torch.tensor([[self.end_of_text, self.end_of_human, self.start_of_ai, self.start_of_speech]], dtype=torch.int64)
        end_speech_tokens = torch.tensor([[self.end_of_speech, self.end_of_ai]], dtype=torch.int64)
        
        # Tokenize reference text
        ref_text_tokens = self.direct_tokenizer(reference_text, return_tensors="pt").input_ids
        
        # Tokenize new text
        new_text_tokens = self.direct_tokenizer(new_text, return_tensors="pt").input_ids
        
        # Create the zero-shot prompt structure following Trelis approach
        # First part: reference text + audio
        first_part = torch.cat([
            start_token, 
            ref_text_tokens, 
            mid_tokens, 
            torch.tensor([reference_audio_tokens], dtype=torch.int64),
            end_speech_tokens
        ], dim=1)
        
        # Second part: new text to be spoken
        second_part = torch.cat([
            start_token,
            new_text_tokens,
            torch.tensor([[self.end_of_text, self.end_of_human, self.start_of_ai, self.start_of_speech]], dtype=torch.int64)
        ], dim=1)
        
        # Combine both parts
        full_prompt_ids = torch.cat([first_part, second_part], dim=1)
        
        # Convert to string for the model
        prompt_string = self.direct_tokenizer.decode(full_prompt_ids[0])
        
        return prompt_string, full_prompt_ids
    
    def clone_voice(self, reference_audio_path, reference_text, new_text, output_path=None, **kwargs):
        """Generate speech in the voice of the reference audio using Trelis approach"""
        logger.info(f"Loading reference audio from {reference_audio_path}")
        
        # Load audio exactly as in Trelis code
        audio_array, sample_rate = librosa.load(reference_audio_path, sr=24000)
        
        # Get SNAC tokens from reference audio
        logger.info("Tokenizing reference audio...")
        audio_tokens = self.tokenise_audio(audio_array, sample_rate)
        if audio_tokens is None or len(audio_tokens) == 0:
            raise ValueError("Failed to extract audio tokens from reference audio")
        
        logger.info(f"Using {len(audio_tokens)} audio tokens from reference audio")
        
        # Format the prompt for zero-shot cloning
        logger.info("Formatting prompt for voice cloning...")
        formatted_prompt, _ = self.format_cloning_prompt(reference_text, audio_tokens, new_text)
        
        # Generate speech with the cloned voice
        logger.info("Generating speech with cloned voice...")
        start_time = time.monotonic()
        
        # Set default generation parameters exactly as in Trelis
        if 'repetition_penalty' not in kwargs:
            kwargs['repetition_penalty'] = 1.1
        if 'stop_token_ids' not in kwargs:
            kwargs['stop_token_ids'] = [self.end_of_speech]
        if 'max_tokens' not in kwargs:
            kwargs['max_tokens'] = 990
        if 'temperature' not in kwargs:
            kwargs['temperature'] = 0.5
        if 'top_p' not in kwargs:
            kwargs['top_p'] = 0.9
        
        # Generate speech
        syn_tokens = self.generate_speech(
            prompt=formatted_prompt,
            **kwargs
        )
        
        # Save audio if output path is provided
        if output_path:
            with wave.open(output_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                
                total_frames = 0
                for audio_chunk in syn_tokens:
                    frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
                    total_frames += frame_count
                    wf.writeframes(audio_chunk)
                
                duration = total_frames / wf.getframerate()
                end_time = time.monotonic()
                logger.info(f"Generated {duration:.2f} seconds of audio in {end_time - start_time:.2f} seconds")
                logger.info(f"Real-time factor: {(end_time - start_time) / duration:.2f}x")
                logger.info(f"Audio saved to {output_path}")
        
        return syn_tokens
    
    def extract_audio_from_generated(self, generated_ids):
        """Extract and process audio tokens from generated IDs"""
        # Find the last occurrence of start_of_speech token
        token_indices = (generated_ids == self.start_of_speech).nonzero(as_tuple=True)
        
        if len(token_indices[1]) > 0:
            last_occurrence_idx = token_indices[1][-1].item()
            # Extract tokens after the start_of_speech token
            audio_tokens = generated_ids[:, last_occurrence_idx+1:]
            
            # Remove end_of_speech token if present
            audio_tokens = audio_tokens[audio_tokens != self.end_of_speech]
            
            # Ensure token count is a multiple of 7
            token_count = audio_tokens.size(0)
            new_length = (token_count // 7) * 7
            audio_tokens = audio_tokens[:new_length]
            
            # Adjust token values for SNAC decoding
            audio_tokens = [t.item() - self.audio_tokens_start for t in audio_tokens]
            
            return audio_tokens
        
        return None
    
    def tokens_to_audio(self, tokens):
        """Convert SNAC tokens back to audio waveform"""
        if not tokens or len(tokens) < 7:
            return None
            
        # Redistribute tokens into SNAC layers
        layer_1 = []
        layer_2 = []
        layer_3 = []
        
        for i in range(len(tokens) // 7):
            layer_1.append(tokens[7*i])
            layer_2.append(tokens[7*i+1] - 4096)
            layer_3.append(tokens[7*i+2] - (2*4096))
            layer_3.append(tokens[7*i+3] - (3*4096))
            layer_2.append(tokens[7*i+4] - (4*4096))
            layer_3.append(tokens[7*i+5] - (5*4096))
            layer_3.append(tokens[7*i+6] - (6*4096))
            
        codes = [
            torch.tensor(layer_1).unsqueeze(0),
            torch.tensor(layer_2).unsqueeze(0),
            torch.tensor(layer_3).unsqueeze(0)
        ]
        
        # Decode audio
        with torch.inference_mode():
            audio = self.snac_model.decode(codes)
            
        return audio

def main():
    parser = argparse.ArgumentParser(description="Voice synthesis with Orpheus TTS - supports both zero-shot cloning and speaker-based generation")
    parser.add_argument("--model", type=str, 
                        default="/vast/audio/experiment/Orpheus-TTS/speaker_finetune/checkpoints/checkpoint-90484", 
                        help="Path to the model checkpoint")
    
    # Common parameters
    parser.add_argument("--text", type=str,
                        default="""يا رجل <ضحكة مكتومة>، التغيير اللي خلّته وسائل التواصل الاجتماعي في طريقة تعاملنا مع بعض شيء جنوني والله! <قهقهة> يعني، إحنا كلنا متصلين 24 ساعة، لكن برضه الناس حسّت نفسها وحيدة أكتر من أي وقتٍ مضى... <تنهيدة> وماتفتحش موضوع تأثيره على تقدير الذات والصحة النفسية عند الأطفال وكده.""", 
                        help="Text to synthesize")
    parser.add_argument("--output", type=str, 
                        default="../cloned_fft.wav", 
                        help="Output audio file path")
    parser.add_argument("--temperature", type=float, 
                        default=0.8, 
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, 
                        default=0.5, 
                        help="Top-p sampling parameter")
    
    # Zero-shot cloning parameters
    parser.add_argument("--reference", type=str,
                        help="Path to reference audio file for zero-shot cloning")
    parser.add_argument("--reference_text", type=str,
                        default="وبدخل انا مع الموتو زي ما يقولوا اذا بدخل في دراجتي الماضية وما سويت الشغلات ه",  
                        help="Transcript of the reference audio")
    
    # Speaker-based generation parameters
    parser.add_argument("--voice", type=str,
                        default="spk_sp_ar_39",
                        help="Voice ID for speaker-based generation")
    
    args = parser.parse_args()
    
    logger.info(f"Loading model from {args.model}")
    model = ZeroShotCloningModel(model_name=args.model)
    
    # Determine which mode to use based on provided arguments
    if args.reference:
        # Zero-shot cloning mode
        logger.info(f"Cloning voice from {args.reference}")
        model.clone_voice(
            reference_audio_path=args.reference,
            reference_text=args.reference_text,
            new_text=args.text,
            output_path=args.output,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=1.3,
            stop_token_ids=[model.end_of_speech],
            max_tokens=8192
        )
    else:
        # Speaker-based generation mode
        logger.info(f"Using speaker-based generation with voice: {args.voice}")
        model.generate_with_voice(
            prompt=args.text,
            voice=args.voice,
            output_path=args.output,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=1.3,
            stop_token_ids=[model.end_of_speech],
            max_tokens=8192
        )

if __name__ == "__main__":
    main()



#/vast/audio/data/tts/22k/ar_spotify_v1/data/asr/workspace/audio/tts/data/tts/ar/spotify/wav/22k/93ebc6ee-f879-465b-acdd-264aa7f6cb75.wav
#وبدخل انا مع الموتو زي ما يقولوا اذا بدخل في دراجتي الماضية وما سويت الشغلات ه