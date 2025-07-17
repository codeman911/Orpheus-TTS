# Orpheus-TTS: Zero-Shot Voice Cloning System

A text-to-speech system capable of zero-shot voice cloning using a reference audio sample.

## Pipeline Overview

### 1. System Architecture

The system consists of three main components:
- Base TTS Model (Orpheus)
- SNAC (Speech Neural Audio Codec) for audio tokenization
- Speaker Conditioning Mechanism

### 2. Data Preparation Pipeline

#### 2.1 Training Data Requirements
- Text-audio paired data
- Multi-speaker dataset
- Audio requirements:
  - Sample rate: 24kHz
  - Format: Mono channel
  - Recommended length: 3-10 seconds per sample

#### 2.2 Data Processing
1. Audio Preprocessing:
   - Resample to 24kHz
   - Convert to mono
   - Normalize audio levels
   - Remove silence/noise

2. Text Preprocessing:
   - Text normalization
   - Special token handling (<laugh>, <sigh>, etc.)
   - Speaker ID tagging

### 3. Training Pipeline

#### 3.1 Speaker Fine-tuning
```python
# Training command example
python train.py \
    --data_path /path/to/processed_data \
    --output_dir ./checkpoints_spk \
    --model_name base_model \
    --batch_size 8 \
    --learning_rate 1e-5
```
