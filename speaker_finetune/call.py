import httpx
import time
import torch
import torchaudio
import numpy as np
import subprocess
import tempfile
import os

API_URL = "http://localhost:8000/generate_audio"


def stream_audio(text, voice):
    with httpx.stream(
        "POST",
        API_URL,
        data={"text": text, "voice": voice},
        headers={"accept": "audio/wav"}
    ) as response:
        if response.status_code != 200:
            print("Failed to get audio stream:", response)
            return

        # Save to temporary file first
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
            
            # Write chunks to file
            for chunk in response.iter_bytes():
                if chunk:
                    temp_file.write(chunk)
        
        try:
            # Use system player directly since torchaudio FFmpeg extension isn't working
            print(f"Playing audio using system player...")
            subprocess.run(["afplay", temp_path])
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass


if __name__ == "__main__":
    arabic_text = """هالزمن تغير وايد، والأيام تمر بسرعة ما نحس فيها.
قبل كنا نقعد مع الأهل والسوالف ما تخلص، كل حد يضحك ويشارك قصصه، أما الحين، كل واحد ماسك تلفونه وما حد يدري عن الثاني.
حتى القهاوي اللي كنا نروحها أول، كانت بسيطة وشاي الكرك هو الأساس، بس الحين الكوفيات صارت كلها ستايل وأسماء غريبة.
بس الصراحة، رغم كل هالتغييرات، يظل الترابط بين الأهل والأصدقاء أهم شي، لأن في النهاية، العزوة والوناسة هي اللي تخلي الحياة لها طعم."""
    arabic_text = "Hello there. Today we're going to discuss a debate that's probably as old as refrigeration itself. Should you keep tomato ketchup in the fridge? I know, I know. It doesn't matter one way or the other."
    selected_voice = "Emirati_female_1"

    stream_audio(arabic_text, selected_voice)
