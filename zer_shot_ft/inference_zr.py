from orpheus_tts import OrpheusModel
import wave
import time
import librosa  
import numpy as np 
import torch
from snac import SNAC
import os

# Initialize model
model = OrpheusModel(model_name="/vast/audio/experiment/Orpheus-TTS/zer_shot_ft/checkpoints_zr/checkpoint-12000")

# Initialize SNAC model for audio encoding
snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
snac_model = snac_model.to("cuda" if torch.cuda.is_available() else "cpu")

def generate_voice_cloning_prompt(reference_audio_path, reference_text, target_text, tokenizer, snac_model, device="cuda"):
    """
    Generate a properly formatted prompt for zero-shot voice cloning.
    
    Args:
        reference_audio_path (str): Path to the reference audio file
        reference_text (str): Transcript of the reference audio
        target_text (str): Text to be synthesized in the cloned voice
        tokenizer: The tokenizer for text encoding
        snac_model: The SNAC model for audio encoding
        device (str): Device to use for processing
        
    Returns:
        torch.Tensor: Formatted input_ids tensor for model inference
    """
    # Load and process reference audio
    audio_array, sample_rate = librosa.load(reference_audio_path, sr=24000)
    
    # Convert to mono if needed
    if len(audio_array.shape) > 1:
        audio_array = np.mean(audio_array, axis=0)
    
    # Tokenize audio using SNAC model
    waveform = torch.from_numpy(audio_array).unsqueeze(0)
    waveform = waveform.to(dtype=torch.float32, device=device)
    waveform = waveform.unsqueeze(0)
    
    with torch.inference_mode():
        codes = snac_model.encode(waveform)
    
    # Process audio codes
    audio_tokens_start = 128266
    reference_codes = []
    for i in range(codes[0].shape[1]):
        reference_codes.append(codes[0][0][i].item() + audio_tokens_start)
        reference_codes.append(codes[1][0][2*i].item() + audio_tokens_start + 4096)
        reference_codes.append(codes[2][0][4*i].item() + audio_tokens_start + (2*4096))
        reference_codes.append(codes[2][0][(4*i)+1].item() + audio_tokens_start + (3*4096))
        reference_codes.append(codes[1][0][(2*i)+1].item() + audio_tokens_start + (4*4096))
        reference_codes.append(codes[2][0][(4*i)+2].item() + audio_tokens_start + (5*4096))
        reference_codes.append(codes[2][0][(4*i)+3].item() + audio_tokens_start + (6*4096))
    
    # Define special tokens
    start_of_human = 128259
    end_of_human = 128260
    start_of_ai = 128261
    start_of_speech = 128257
    end_of_speech = 128258
    end_of_ai = 128262
    end_of_text = 128009
    
    # Tokenize reference and target text
    reference_text_ids = tokenizer.encode(reference_text, add_special_tokens=False)
    target_text_ids = tokenizer.encode(target_text, add_special_tokens=False)
    
    # Create the prompt structure following zero_shot_datasets.py format
    # Format: SOH + ref_text + EOT + EOH + SOA + SOS + audio_codes + EOS + EOA + SOH + target_text + EOT + EOH
    prompt_ids = (
        [start_of_human] 
        + reference_text_ids 
        + [end_of_text, end_of_human]
        + [start_of_ai] 
        + [start_of_speech] 
        + reference_codes 
        + [end_of_speech] 
        + [end_of_ai]
        + [start_of_human] 
        + target_text_ids 
        + [end_of_text, end_of_human]
    )
    
    # Convert to tensor
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long)
    
    return prompt_tensor

def clone_voice_and_generate(reference_audio_path, reference_text, target_text, output_path="output.wav"):
    """
    Clone a voice from reference audio and generate speech with the target text
    
    Args:
        reference_audio_path (str): Path to reference audio file
        reference_text (str): Transcript of the reference audio
        target_text (str): Text to synthesize in the cloned voice
        output_path (str): Path to save the generated audio
    """
    # Import the tokenizer directly from transformers
    from transformers import AutoTokenizer
    
    # Load the tokenizer from the same model path
    tokenizer = AutoTokenizer.from_pretrained(model.model_name)
    
    # Generate the voice cloning prompt
    prompt_tensor = generate_voice_cloning_prompt(
        reference_audio_path=reference_audio_path,
        reference_text=reference_text,
        target_text=target_text,
        tokenizer=tokenizer,
        snac_model=snac_model
    )
    
    # Instead of passing token IDs directly, we need to use the reference_audio parameter
    # which is designed for voice cloning in the OrpheusModel
    start_time = time.monotonic()
    
    # Use the model's built-in voice cloning functionality
    syn_tokens = model.generate_speech(
        prompt=target_text,  # Pass the target text as a string
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.4,
        max_tokens=1200,
        reference_audio=reference_audio_path,  # Pass the reference audio path
        reference_text=reference_text  # Pass the reference text
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Save the generated audio
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        
        total_frames = 0
        chunk_counter = 0
        for audio_chunk in syn_tokens:  # output streaming
            chunk_counter += 1
            frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
            total_frames += frame_count
            wf.writeframes(audio_chunk)
        duration = total_frames / wf.getframerate()
        
        end_time = time.monotonic()
        print(f"It took {end_time - start_time:.2f} seconds to generate {duration:.2f} seconds of audio")
        print(f"Audio saved to {output_path}")
    
    return output_path

# Example usage
if __name__ == "__main__":
    reference_audio = "../male_6.wav"
    reference_text = """ويمكن موضوع الملوحة في الماء. بس أنت يعني تعطينا Tips في صناعة المحتوى الزراعي عن كيف أننا نستعمل هذه التقنيات وكيف نستعمل، كيف نتعامل مع الفشل في الزراعة. شو ممكن تخبرنا عن صناعة المحتوى؟"""
    target_text = "I finally got into the university of my dreams! I can't believe all this hard work actually paid off!"
    
    output_path = "/Users/vikram.solanki/Orpheus-TTS/generated_audio/cloned_voice.wav"
    clone_voice_and_generate(reference_audio, reference_text, target_text, output_path)