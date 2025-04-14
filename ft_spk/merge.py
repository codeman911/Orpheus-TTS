import os
import shutil
from datasets import Dataset, concatenate_datasets
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def merge_dataset_parts(parts_dir, output_dir):
    """Merge dataset parts into a single dataset"""
    
    # Get all part directories
    part_dirs = [d for d in os.listdir(parts_dir) if d.startswith("part_") and os.path.isdir(os.path.join(parts_dir, d))]
    part_dirs.sort(key=lambda x: int(x.split("_")[1]))
    
    logger.info(f"Found {len(part_dirs)} parts to merge")
    
    # Load each part
    all_parts = []
    for part_dir in part_dirs:
        part_path = os.path.join(parts_dir, part_dir)
        try:
            logger.info(f"Loading {part_dir} from {part_path}")
            part_dataset = Dataset.load_from_disk(part_path)
            all_parts.append(part_dataset)
            logger.info(f"{part_dir} loaded with {len(part_dataset)} examples")
        except Exception as e:
            logger.error(f"Error loading {part_dir}: {str(e)}")
    
    # Merge parts
    if all_parts:
        logger.info(f"Concatenating {len(all_parts)} parts...")
        final_dataset = concatenate_datasets(all_parts)
        
        # Save the final dataset
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving final dataset with {len(final_dataset)} examples to {output_dir}")
        final_dataset.save_to_disk(output_dir)
        
        # Verify the final dataset
        try:
            verification = Dataset.load_from_disk(output_dir)
            logger.info(f"Final dataset verified with {len(verification)} examples")
        except Exception as e:
            logger.error(f"Error verifying final dataset: {str(e)}")
        
        return final_dataset
    else:
        logger.error("No parts were successfully loaded for merging")
        return None

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Merge dataset parts")
    parser.add_argument("--parts-dir", type=str, required=True, help="Directory containing dataset parts")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for merged dataset")
    
    args = parser.parse_args()
    
    merge_dataset_parts(args.parts_dir, args.output_dir)