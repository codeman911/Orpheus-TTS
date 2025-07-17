import os
import torch
import torchaudio
import argparse
import yaml
import logging
import numpy as np
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC
import torchaudio.transforms as T

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

snac_model = None
tokenizer = None
model = None
config = None

def set_random_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def load_models(snac_model_path):
    global model, tokenizer, snac_model, config
    model_path = "shahink/tts_v2_st"
    logger.info(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=2048,
        padding_side="left",
        use_fast=True,
        legacy=True
    )
    logger.info(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model = model.to("cuda:0")
    model.eval()
    logger.info(f"Loading SNAC model from {snac_model_path}")
    snac_model = SNAC.from_pretrained(snac_model_path).to("cuda:0")
    snac_model.eval()

def tokenise_audio(waveform, sample_rate=24000, audio_tokens_start=128266):
    global snac_model
    try:
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform).unsqueeze(0)
        waveform = waveform.to(dtype=torch.float32)
        if sample_rate != 24000:
            resample_transform = T.Resample(orig_freq=sample_rate, new_freq=24000)
            waveform = resample_transform(waveform)
        waveform = waveform.unsqueeze(0).to("cuda:0")
        with torch.inference_mode():
            codes = snac_model.encode(waveform)
        all_codes = []
        for i in range(codes[0].shape[1]):
            all_codes.append(codes[0][0][i].item() + audio_tokens_start)
            all_codes.append(codes[1][0][2*i].item() + audio_tokens_start + 4096)
            all_codes.append(codes[2][0][4*i].item() + audio_tokens_start + (2*4096))
            all_codes.append(codes[2][0][(4*i)+1].item() + audio_tokens_start + (3*4096))
            all_codes.append(codes[1][0][(2*i)+1].item() + audio_tokens_start + (4*4096))
            all_codes.append(codes[2][0][(4*i)+2].item() + audio_tokens_start + (5*4096))
            all_codes.append(codes[2][0][(4*i)+3].item() + audio_tokens_start + (6*4096))
        return all_codes
    except Exception as e:
        logger.error(f"Error in tokenise_audio: {str(e)}")
        return None

def remove_duplicate_frames(codes_list):
    if codes_list is None or len(codes_list) == 0:
        return None
    if len(codes_list) % 7 != 0:
        codes_list = codes_list[:-(len(codes_list) % 7)]
    if len(codes_list) == 0:
        return None
    result = codes_list[:7]
    for i in range(7, len(codes_list), 7):
        if i+6 >= len(codes_list):
            break
        current_first = codes_list[i]
        previous_first = result[-7]
        if current_first != previous_first:
            result.extend(codes_list[i:i+7])
    if len(result) < 7:
        return None
    return result

def decode_audio_tokens(audio_tokens, audio_tokens_start=128266):
    global snac_model
    if not audio_tokens or len(audio_tokens) < 7:
        logger.error("Not enough audio tokens to decode")
        return None
    if len(audio_tokens) % 7 != 0:
        audio_tokens = audio_tokens[:-(len(audio_tokens) % 7)]
    level_0_tokens = []
    level_1_tokens = []
    level_2_tokens = []
    for i in range(0, len(audio_tokens), 7):
        level_0_tokens.append(audio_tokens[i] - audio_tokens_start)
        level_1_tokens.extend([
            audio_tokens[i+1] - (audio_tokens_start + 4096),
            audio_tokens[i+4] - (audio_tokens_start + 4*4096)
        ])
        level_2_tokens.extend([
            audio_tokens[i+2] - (audio_tokens_start + 2*4096),
            audio_tokens[i+3] - (audio_tokens_start + 3*4096),
            audio_tokens[i+5] - (audio_tokens_start + 5*4096),
            audio_tokens[i+6] - (audio_tokens_start + 6*4096)
        ])
    level_0 = torch.tensor(level_0_tokens, dtype=torch.long).unsqueeze(0).to("cuda:0")
    level_1 = torch.tensor(level_1_tokens, dtype=torch.long).unsqueeze(0).to("cuda:0")
    level_2 = torch.tensor(level_2_tokens, dtype=torch.long).unsqueeze(0).to("cuda:0")
    with torch.inference_mode():
        waveform = snac_model.decode([level_0, level_1, level_2])
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() == 3:
            waveform = waveform.squeeze(0)
    return waveform.cpu()

def generate_speech_for_chunk(reference_codes, reference_text, text_chunk, output_path, max_new_tokens=2000):
    global model, tokenizer, config
    set_random_seed(42)
    generation_config = {
        'temperature': 0.3,
        'top_p': 0.95,
        'repetition_penalty': 1.1,
        'max_new_tokens': max_new_tokens,
        'do_sample': True,
        'return_dict_in_generate': True,
        'output_scores': True
    }
    start_of_human = config.get("start_of_human", 128259)
    end_of_human = config.get("end_of_human", 128260)
    start_of_ai = config.get("start_of_ai", 128261)
    start_of_speech = config.get("start_of_speech", 128257)
    end_of_speech = config.get("end_of_speech", 128258)
    end_of_ai = config.get("end_of_ai", 128262)
    end_of_text = config.get("end_of_text", 128009)
    reference_text_ids = tokenizer.encode(reference_text, add_special_tokens=True)
    target_text_ids = tokenizer.encode(text_chunk, add_special_tokens=True)
    input_ids = (
        [start_of_human] + reference_text_ids + [end_of_text, end_of_human] +
        [start_of_ai] + [start_of_speech] + reference_codes + [end_of_speech] + [end_of_ai] +
        [start_of_human] + target_text_ids + [end_of_text, end_of_human] +
        [start_of_ai] + [start_of_speech]
    )
    input_tensor = torch.tensor([input_ids], device="cuda:0")
    with torch.inference_mode():
        outputs = model.generate(
            input_tensor,
            **generation_config
        )
        new_tokens = outputs.sequences[0, len(input_ids):].tolist()
    audio_tokens = [t for t in new_tokens if t >= 128266][:7*1024]
    waveform = decode_audio_tokens(audio_tokens)
    if waveform is not None:
        torchaudio.save(output_path, waveform, 24000)
        logger.info(f"Saved audio chunk to {output_path}")
        return output_path
    return None

def split_text_into_sentences(text):
    words = text.strip().split()
    if len(words) <= 5:
        return [' '.join(words)]
    first_chunk = ' '.join(words[:5])
    rest_text = ' '.join(words[5:])
    # Split the rest by period or comma, keeping the delimiter
    import re
    sentences = re.split(r'(?<=[.,])\s+', rest_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return [first_chunk] + sentences

def main():
    parser = argparse.ArgumentParser(description="Zero-shot TTS inference with state preservation")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--reference-audio", type=str, required=True, help="Path to reference audio file")
    parser.add_argument("--reference-text", type=str, default=""""في مستقبلٍ قريب، سيصبح الذكاء الاصطناعي القوة الدافعة التي تُعيد تشكيل عالمنا، حيث يمتزج الإبداع البشري مع قوة الخوارزميات لخلق حلولٍ غير مسبوقة. سيطور الذكاء الاصطناعي قدراتٍ تفوق الخيال، من تشخيص الأمراض بدقةٍ فائقة إلى تصميم مدنٍ ذكيةٍ تعمل بتناغمٍ تام، مما يفتح آفاقاً جديدةً للتقدم والرفاهية. لكن هذا التطور يحمل أيضاً تحدياتٍ أخلاقيةً عميقة، فالتوازن بين التحكم البشري والاستقلالية الآلية سيكون مفتاحاً لمستقبلٍ يعم فيه العدل والاستدامة." """, help="Reference text matching the audio")
    parser.add_argument("--target-text", type=str, default=""""Earlier on Thursday. he confirmed that there was a. "direct hit to multiple homes", and that no one in those homes were believed to be seriously injured.Footage from the scene shows the charred cars littered across the street.Local resident Christopher Moore told the Associated Press that he and his wife were woken by a loud bang in the early hours of the morning.""", help="Text to synthesize in the reference voice")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save output audio files")
    parser.add_argument("--max-new-tokens", type=int, default=2000, help="Maximum number of new tokens to generate per sentence")
    args = parser.parse_args()
    global config
    config = load_config(args.config)
    snac_model_path = config.get("snac_model", "hubertsiuzdak/snac_24khz")
    load_models(snac_model_path)
    logger.info(f"Loading reference audio from {args.reference_audio}")
    waveform, sample_rate = torchaudio.load(args.reference_audio)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    logger.info("Tokenizing reference audio")
    audio_tokens_start = config.get("audio_tokens_start", 128266)
    reference_codes = tokenise_audio(waveform, sample_rate, audio_tokens_start)
    if reference_codes is None or len(reference_codes) == 0:
        logger.error("Failed to tokenize reference audio")
        return
    logger.info(f"Reference codes length: {len(reference_codes)}")
    logger.info(f"First 10 reference codes: {reference_codes[:10]}")
    reference_codes = remove_duplicate_frames(reference_codes)
    if reference_codes is None or len(reference_codes) == 0:
        logger.error("No valid frames in reference audio")
        return
    sentences = split_text_into_sentences(args.target_text)
    logger.info(f"Split target text into {len(sentences)} sentences")
    os.makedirs(args.output_dir, exist_ok=True)
    results = []
    for idx, sentence in enumerate(sentences):
        output_path = os.path.join(args.output_dir, f"chunk_{idx+1}.wav")
        logger.info(f"Generating audio for chunk {idx+1}: {sentence}")
        result = generate_speech_for_chunk(
            reference_codes,
            args.reference_text,
            sentence,
            output_path,
            args.max_new_tokens
        )
        if result:
            results.append(result)
    logger.info(f"Successfully generated {len(results)} audio files in {args.output_dir}")

if __name__ == "__main__":
    main()