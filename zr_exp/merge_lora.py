import os
import torch
import yaml
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def merge_lora_checkpoint(
    base_model_path: str,
    lora_adapter_path: str,
    output_path: str,
    config_path: str = "config.yaml"
):
    """
    Merge LoRA adapter with base model and save as full checkpoint.
    
    Args:
        base_model_path: Path to the base model
        lora_adapter_path: Path to the LoRA adapter weights
        output_path: Path where to save the merged model
        config_path: Path to config file for model settings
    """
    
    # Load config if available
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {config_path}")
    
    try:
        logger.info(f"Loading base model from: {base_model_path}")
        
        # Load base model with same settings as training
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        logger.info(f"Loading LoRA adapter from: {lora_adapter_path}")
        
        # Load the model with LoRA adapter
        model = PeftModel.from_pretrained(
            base_model,
            lora_adapter_path,
            torch_dtype=torch.bfloat16
        )
        
        logger.info("Merging LoRA adapter with base model...")
        
        # Merge the adapter into the base model
        merged_model = model.merge_and_unload()
        
        # Load tokenizer from base model (checkpoints don't contain tokenizer)
        logger.info(f"Loading tokenizer from base model: {base_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Save merged model
        logger.info(f"Saving merged model to: {output_path}")
        merged_model.save_pretrained(
            output_path,
            save_function=torch.save,
            safe_serialization=True
        )
        
        # Save tokenizer
        logger.info(f"Saving tokenizer to: {output_path}")
        tokenizer.save_pretrained(output_path)
        
        # Save model info
        model_info = {
            "base_model": base_model_path,
            "lora_adapter": lora_adapter_path,
            "merged_at": str(torch.utils.data.get_worker_info()),
            "torch_dtype": "bfloat16",
            "model_type": "merged_lora_checkpoint"
        }
        
        import json
        with open(os.path.join(output_path, "merge_info.json"), "w") as f:
            json.dump(model_info, f, indent=2)
        
        logger.info("✅ Successfully merged and saved checkpoint!")
        logger.info(f"📁 Full checkpoint saved to: {output_path}")
        
        # Print model size info
        param_count = sum(p.numel() for p in merged_model.parameters())
        logger.info(f"📊 Merged model parameters: {param_count:,}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during merging: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument(
        "--base-model", 
        type=str, 
        required=True,
        help="Path to base model (same as used in training)"
    )
    parser.add_argument(
        "--lora-adapter", 
        type=str, 
        required=True,
        help="Path to LoRA adapter weights directory"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        required=True,
        help="Output path for merged checkpoint"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml",
        help="Path to config file"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.lora_adapter):
        logger.error(f"LoRA adapter path does not exist: {args.lora_adapter}")
        return
    
    # Check for adapter files
    adapter_files = ["adapter_model.safetensors", "adapter_model.bin"]
    found_adapter = any(os.path.exists(os.path.join(args.lora_adapter, f)) for f in adapter_files)
    
    if not found_adapter:
        logger.error(f"No LoRA adapter files found in: {args.lora_adapter}")
        logger.error(f"Expected one of: {', '.join(adapter_files)}")
        logger.error("Available files:")
        try:
            files = os.listdir(args.lora_adapter)
            for f in files:
                logger.error(f"  - {f}")
        except:
            pass
        return
    
    # Check for adapter config
    if not os.path.exists(os.path.join(args.lora_adapter, "adapter_config.json")):
        logger.error(f"Missing adapter_config.json in: {args.lora_adapter}")
        return
    
    # Perform merge
    success = merge_lora_checkpoint(
        base_model_path=args.base_model,
        lora_adapter_path=args.lora_adapter,
        output_path=args.output,
        config_path=args.config
    )
    
    if success:
        logger.info("🎉 Merge completed successfully!")
    else:
        logger.error("💥 Merge failed!")
        exit(1)

# Convenience function for quick merging with default paths
def quick_merge_from_config():
    """Quick merge using paths from config.yaml"""
    
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        base_model = config["model_name"]
        save_folder = config["save_folder"]
        
        lora_adapter_path = f"./{save_folder}/lora_adapter"
        output_path = f"./{save_folder}/merged_checkpoint"
        
        logger.info("🚀 Quick merge using config.yaml paths:")
        logger.info(f"   Base model: {base_model}")
        logger.info(f"   LoRA adapter: {lora_adapter_path}")
        logger.info(f"   Output: {output_path}")
        
        if not os.path.exists(lora_adapter_path):
            logger.error(f"LoRA adapter not found at: {lora_adapter_path}")
            logger.error("Make sure training has completed and saved the adapter")
            return False
        
        return merge_lora_checkpoint(
            base_model_path=base_model,
            lora_adapter_path=lora_adapter_path,
            output_path=output_path
        )
        
    except Exception as e:
        logger.error(f"Error in quick merge: {str(e)}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("🔄 LoRA Checkpoint Merger")
    print("="*60)
    
    # Check if being run with arguments
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        # Try quick merge from config
        logger.info("No arguments provided, attempting quick merge from config.yaml...")
        success = quick_merge_from_config()
        
        if not success:
            print("\n" + "="*60)
            print("📖 Usage Examples:")
            print("="*60)
            print("1. Quick merge (uses config.yaml):")
            print("   python merge_lora_checkpoint.py")
            print("")
            print("2. Custom paths:")
            print("   python merge_lora_checkpoint.py \\")
            print("     --base-model v1kram/spk_ft \\")
            print("     --lora-adapter ./checkpoints_spk/lora_adapter \\")
            print("     --output ./checkpoints_spk/merged_checkpoint")
            print("")
            print("3. With custom config:")
            print("   python merge_lora_checkpoint.py \\")
            print("     --base-model v1kram/spk_ft \\")
            print("     --lora-adapter ./checkpoints_spk/lora_adapter \\")
            print("     --output ./merged_model \\")
            print("     --config my_config.yaml")
            exit(1)
        else:
            logger.info("🎉 Quick merge completed successfully!") 
