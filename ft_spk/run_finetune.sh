#!/bin/bash
# Script to prepare local dataset and run fine-tuning

# Function to extract value from YAML
extract_yaml_value() {
  local file="$1"
  local key="$2"
  # Extract value using grep and sed
  local value=$(grep -E "^$key:" "$file" | sed -E "s/^$key: \"?([^\"]+)\"?/\1/")
  echo "$value"
}

# Default values
MANIFEST_PATH="/vast/audio/experiment/Orpheus-TTS/datasets/train_sample.json"

# Read dataset path from config.yaml first
CONFIG_FILE="config.yaml"
if [ -f "$CONFIG_FILE" ]; then
  DATASET_DIR=$(extract_yaml_value "$CONFIG_FILE" "TTS_dataset")
  # Remove trailing slash if present for consistency
  DATASET_DIR=${DATASET_DIR%/}
  echo "Read dataset directory from config: $DATASET_DIR"
else
  # Default if config not found
  DATASET_DIR="/vast/audio/data/tts/oprphs_data"
  echo "Config file not found, using default dataset path: $DATASET_DIR"
fi

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --manifest)
      MANIFEST_PATH="$2"
      shift 2
      ;;
    --dataset-dir)
      DATASET_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check if manifest path is provided
if [ -z "$MANIFEST_PATH" ]; then
  echo "Error: Manifest path is required. Use --manifest to specify the path."
  echo "Example: ./run_finetune.sh --manifest /path/to/your/manifest.jsonl"
  exit 1
fi

# Always update config with the current dataset directory to ensure consistency
# Use sed to update the dataset path in config.yaml
sed -i '' "s|TTS_dataset: \".*\"|TTS_dataset: \"$DATASET_DIR\"|g" config.yaml
echo "Ensured config.yaml is using dataset directory: $DATASET_DIR"

# Step 1: Prepare the dataset
echo "Preparing dataset from manifest: $MANIFEST_PATH"

# Try python3 first, then fall back to python if needed
if command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD="python"
else
  echo "Error: Neither python3 nor python is available in your PATH."
  echo "Please make sure Python is installed and available in your PATH."
  exit 1
fi

echo "Using Python command: $PYTHON_CMD"
$PYTHON_CMD prepare_local_dataset.py --manifest "$MANIFEST_PATH" --output "$DATASET_DIR"

# Check if dataset preparation was successful
if [ $? -ne 0 ]; then
  echo "Error: Dataset preparation failed."
  exit 1
fi

# Step 2: Run fine-tuning
echo "Starting fine-tuning..."

# Check if accelerate is available
if command -v accelerate >/dev/null 2>&1; then
  echo "Using accelerate for distributed training"
  accelerate launch train.py
else
  echo "Accelerate not found, attempting to run with $PYTHON_CMD directly"
  $PYTHON_CMD train.py
fi

echo "Fine-tuning process completed."
