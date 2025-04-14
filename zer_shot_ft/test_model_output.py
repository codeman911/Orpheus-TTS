import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def process_input_sequence(tokenizer, reference_text, target_text, config):
    """Process and validate input sequence"""
    logger.info("Processing input sequence...")
    
    # Validate texts
    if not reference_text or not target_text:
        raise ValueError("Reference and target texts cannot be empty")
    
    # Get special tokens
    start_of_human = config.get("start_of_human", 128259)
    end_of_human = config.get("end_of_human", 128260)
    start_of_ai = config.get("start_of_ai", 128261)
    end_of_text = config.get("end_of_text", 128009)
    
    # Encode and validate texts
    try:
        reference_text_ids = tokenizer.encode(reference_text, add_special_tokens=True)
        target_text_ids = tokenizer.encode(target_text, add_special_tokens=True)
        
        logger.info(f"Reference text tokens: {len(reference_text_ids)}")
        logger.info(f"Target text tokens: {len(target_text_ids)}")
        
        # Check token lengths
        if len(reference_text_ids) > 1024 or len(target_text_ids) > 1024:
            logger.warning("Text length exceeds recommended limit of 1024 tokens")
    
    except Exception as e:
        logger.error(f"Error encoding text: {e}")
        raise
    
    # Create and validate input sequence
    input_ids = (
        [start_of_human] 
        + reference_text_ids 
        + [end_of_text, end_of_human]
        + [start_of_ai]
        + [start_of_human] 
        + target_text_ids 
        + [end_of_text, end_of_human]
        + [start_of_ai]
    )
    
    logger.info(f"Total input sequence length: {len(input_ids)}")
    return input_ids

def test_model_output(
    model_path="/vast/audio/experiment/Orpheus-TTS/zer_shot_ft/checkpoints_zr/checkpoint-14720",
    reference_text="""اه يعني رياضة أو شيء...""",
    target_text="""يا رجل <ضحكة مكتومة>...""",
    max_new_tokens=10000
):
    config = load_config()
    
    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=2048,
        padding_side="left",
        use_fast=True,
        legacy=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to("cuda:0")
    model.eval()
    
    # Process input sequence
    try:
        input_ids = process_input_sequence(tokenizer, reference_text, target_text, config)
        input_tensor = torch.tensor([input_ids], device=model.device)
    except Exception as e:
        logger.error(f"Failed to process input sequence: {e}")
        return
    
    # Generate and analyze
    logger.info("Generating output...")
    with torch.inference_mode():
        output = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.3,
            pad_token_id=config.get("pad_token", 128263)
        )
    
    generated_tokens = output[0][len(input_ids):].tolist()
    
    # Analysis
    logger.info("\n=== Model Output Analysis ===")
    logger.info(f"Input sequence length: {len(input_ids)}")
    logger.info(f"Generated sequence length: {len(generated_tokens)}")
    
    try:
        logger.info("\nFirst 50 tokens decoded:")
        logger.info(tokenizer.decode(generated_tokens[:50]))
        
        logger.info("\nLast 50 tokens decoded:")
        logger.info(tokenizer.decode(generated_tokens[-50:]))
        
        # Count special tokens
        special_tokens = {
            'start_of_human': start_of_human,
            'end_of_human': end_of_human,
            'start_of_ai': start_of_ai,
            'end_of_text': end_of_text
        }
        
        logger.info("\nSpecial token counts:")
        for name, token_id in special_tokens.items():
            count = generated_tokens.count(token_id)
            logger.info(f"{name} (ID: {token_id}): {count} occurrences")
            
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
    
    logger.info("=== End Analysis ===")

if __name__ == "__main__":
    test_model_output()