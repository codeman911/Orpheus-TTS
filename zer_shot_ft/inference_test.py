import re
from orpheus_tts import OrpheusModel
import wave
import time
import torch
from transformers import AutoTokenizer

class CustomOrpheusModel(OrpheusModel):
    def __init__(self, model_name, dtype=torch.bfloat16):
        super().__init__(model_name, dtype)
        # Add your custom voices to the available voices list
        self.available_voices.extend([
            "Elise", "8OqhOrMyhp0_SPEAKER_01_0", "1bBRyX90RAw_SPEAKER_05_0","spk_ar_rf_1548","spk_sp_ar_7"
        ])
        print(f"Available voices: {self.available_voices}")
        
        # Override validation to be more permissive
        self._original_validate = self.validate_voice
        self.validate_voice = lambda voice: None
        
        # Load the tokenizer directly to ensure we have access to it
        self.direct_tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def format_prompt_exactly(self, prompt, voice):
        """
        Format the prompt exactly as in the original OrpheusModel._format_prompt method
        This ensures the model sees the same format during inference as during training
        """
        # Format exactly as in the original code (engine_class.py line 61)
        adapted_prompt = f"{voice}: {prompt}"
        
        # Encode text with speaker information
        prompt_tokens = self.direct_tokenizer(adapted_prompt, return_tensors="pt")
        
        # Add special tokens exactly as in the original code
        start_token = torch.tensor([[ 128259]], dtype=torch.int64)
        end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
        
        # Combine tokens
        all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
        
        # Convert to string for the model
        prompt_string = self.direct_tokenizer.decode(all_input_ids[0])
        return prompt_string
    
    def generate_speech_with_exact_format(self, prompt, voice, **kwargs):
        """Generate speech with exact prompt formatting to match training"""
        # Format the prompt using our exact method
        formatted_prompt = self.format_prompt_exactly(prompt, voice)
        
        # Pass the formatted prompt directly to generate_speech
        # Don't pass voice again since it's already in the formatted prompt
        return self.generate_speech(prompt=formatted_prompt, **kwargs)

# Use the custom model
model = CustomOrpheusModel(model_name="/vast/audio/experiment/Orpheus-TTS/speaker_finetune/checkpoints/checkpoint-90484")

# Your prompt with Arabic text and emotion tags
prompt = """يا رجل <ضحكة مكتومة>، التغيير اللي خلّته وسائل التواصل الاجتماعي في طريقة تعاملنا مع بعض شيء جنوني والله! <قهقهة> يعني، إحنا كلنا متصلين 24 ساعة، لكن برضه الناس حسّت نفسها وحيدة أكتر من أي وقتٍ مضى... <تنهيدة> وماتفتحش موضوع تأثيره على تقدير الذات والصحة النفسية عند الأطفال وكده."""
# prompt = '''Man, the way social media has, um, completely changed how we interact is just wild, right? Like, we're all connected 24/7 but somehow people feel more alone than ever. And don't even get me started on how it's messing with kids' self-esteem and mental health and whatnot.'''
# Select voice - must match exactly how it appears in training data
voice = "spk_sp_ar_7"

start_time = time.monotonic()

# Use the new method that formats the prompt exactly like in training
syn_tokens = model.generate_speech_with_exact_format(
    prompt=prompt,
    voice=voice,
    repetition_penalty=1.3,
    stop_token_ids=[128258],  # Explicitly set end_of_speech token
    max_tokens=8192,
    temperature=0.7,
    top_p=0.9
)

with wave.open("../output.wav", "wb") as wf:
   wf.setnchannels(1)
   wf.setsampwidth(2)
   wf.setframerate(24000)

   total_frames = 0
   chunk_counter = 0
   for audio_chunk in syn_tokens:
      chunk_counter += 1
      frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
      total_frames += frame_count
      wf.writeframes(audio_chunk)
   
   duration = total_frames / wf.getframerate()
   end_time = time.monotonic()
   print(f"Generated {duration:.2f} seconds of audio in {end_time - start_time:.2f} seconds")
   print(f"Real-time factor: {(end_time - start_time) / duration:.2f}x")
