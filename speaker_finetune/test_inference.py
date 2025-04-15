import re
import os
from orpheus_tts import OrpheusModel
import wave
import time
import torch
from transformers import AutoTokenizer
import logging

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
class CustomOrpheusModel(OrpheusModel):
    def __init__(self, model_name, dtype=torch.bfloat16):
        super().__init__(model_name, dtype)
        # Add your custom voices to the available voices list
        # self.available_voices.extend([
        #     "Elise", "8OqhOrMyhp0_SPEAKER_01_0", "1bBRyX90RAw_SPEAKER_05_0","spk_ar_rf_1548","spk_sp_ar_7","Emirati_female_1",
        #     "spk_ar_el_v3_13","speaker_ex03","Emirati_male_6","tommy_en_vo","speaker_ex04","speaker_ex01","speaker_ex02","Emirati_580",
        #     "Nouran","spk_sp_en_2","Emirati_528","Emirati_10"
        # ])
        # print(f"Available voices: {self.available_voices}")
        
        # Override validation to be more permissive
        # self._original_validate = self.validate_voice
        # self.validate_voice = lambda voice: None
        
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
        
        # Log the formatted prompt and tokens
        logger.info("\n=== Model Input Analysis ===")
        logger.info(f"Formatted prompt: {formatted_prompt[:100]}...")
        input_tokens = self.direct_tokenizer(formatted_prompt, return_tensors="pt").input_ids
        logger.info(f"Input token shape: {input_tokens.shape}")
        logger.info(f"First 50 input tokens: {input_tokens[0][:50].tolist()}")
        
        # Pass the formatted prompt directly to generate_speech
        output = self.generate_speech(prompt=formatted_prompt, **kwargs)
        
        # Log generation output
        logger.info("\n=== Model Output Analysis ===")
        logger.info(f"Generated output type: {type(output)}")
        if hasattr(output, 'shape'):
            logger.info(f"Output shape: {output.shape}")
        logger.info("=== End Analysis ===\n")
        
        return output

# Use the custom model
model = CustomOrpheusModel(model_name="/vast/audio/experiment/Orpheus-TTS/speaker_finetune/checkpoints_spk/checkpoint-15000")

# Your prompt with Arabic text and emotion tags
# prompt = """"يا رجل، الطريقة اللي غيرت بيها السوشيال ميديا طريقة تفاعلنا مع بعض... والله إنها شيء مجنون، صح؟ يعني إحنا كلنا متصلين 24/7، لكن لسة الناس حاسة نفسها وحيدة أكثر من أي وقت مضى. <laugh> ولا حتى تبدأ تتكلم عن تأثيرها على ثقة الأطفال واحترامهم لذاتهم والصحة النفسية وكذا."""
prompt = '''Man, the way social media has, um, completely changed how we interact is just wild, right? <sigh> Like, we’re all connected 24/7 but somehow people feel more alone than ever. <laugh> And don’t even get me started on how it’s messing with kids’ self-esteem  and mental health and whatnot.'''
# prompt= """यार, सोशल मीडिया ने हमारे रिश्तों को जिस तरह पूरी तरह बदल दिया है, वो सचमुच कितना अजीब है न? मतलब, हम चौबीसों घंटे जुड़े रहते हैं लेकिन फिर भी लोग पहले से ज़्यादा अकेला महसूस करते हैं। और बच्चों के आत्म-विश्वास और मानसिक सेहत पर पड़ रहे बुरे असर की तो बात ही मत छेड़ो... सच कहूँ तो ये सब देखकर डर लगता है।"""
# Select voice - must match exactly how it appears in training data
voice = "Raina"

start_time = time.monotonic()

# Use the new method that formats the prompt exactly like in training
syn_tokens = model.generate_speech_with_exact_format(
    prompt=prompt,
    voice=voice,
    repetition_penalty=1.3,
    stop_token_ids=[128258],  # Explicitly set end_of_speech token
    max_tokens=8192,
    temperature=0.7,
    top_p=0.95
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

# Set up logging at the top of the file after imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
