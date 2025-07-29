from datasets import load_dataset
from datasets import load_from_disk
from datasets import concatenate_datasets
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
import numpy as np
import yaml
import wandb
import torch
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set environment variable for memory management - limit allocation size to 128MB
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True,garbage_collection_threshold:0.6"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,garbage_collection_threshold:0.6"


config_file = "config.yaml"

with open(config_file, "r") as file:
    config = yaml.safe_load(file)

# Keep dataset loading from list of paths
dataset_paths = config["TTS_datasets"]
all_datasets = []

for dataset_path in dataset_paths:
    try:
        logger.info(f"Loading dataset from {dataset_path}")
        dataset = load_from_disk(dataset_path)
        all_datasets.append(dataset)
        logger.info(f"Successfully loaded {len(dataset)} examples from {dataset_path}")
    except Exception as e:
        logger.warning(f"Error loading dataset from {dataset_path}: {str(e)}")
        continue

if not all_datasets:
    raise ValueError("No datasets were successfully loaded!")

# Combine all datasets
ds = concatenate_datasets(all_datasets)
logger.info(f"Combined dataset has {len(ds)} examples")

# Original code implementation from here
model_name = config["model_name"]
run_name = config["run_name"]
project_name = config["project_name"]
base_repo_id = config["save_folder"]
epochs = config["epochs"]
batch_size = config["batch_size"]
save_steps = config["save_steps"]
pad_token = config["pad_token"]
number_processes = config["number_processes"]
learning_rate = config["learning_rate"]
print(f"Using learning rate: {learning_rate}")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="flash_attention_2",torch_dtype=torch.bfloat16)

# Data collator from pretrain/train.py
def data_collator(features):
    # Get max sequence length from config
    max_length = config.get("max_sequence_length", 4096)
    
    # Extract and truncate sequences
    input_ids = [f["input_ids"][:max_length] for f in features]
    
    # Create attention masks
    if any("attention_mask" not in f for f in features):
        attention_mask = [[1] * len(ids) for ids in input_ids]
    else:
        attention_mask = [f["attention_mask"][:max_length] for f in features]
    
    # Create labels - ensure they're the same length as input_ids
    if any("labels" not in f for f in features):
        # Create a deep copy to avoid reference issues
        labels = [ids.copy() for ids in input_ids]
    else:
        # Make sure labels are truncated to the same length as input_ids
        labels = []
        for i, f in enumerate(features):
            if len(f["labels"]) > len(input_ids[i]):
                # Truncate labels to match input_ids length
                labels.append(f["labels"][:len(input_ids[i])])
            elif len(f["labels"]) < len(input_ids[i]):
                # Pad labels to match input_ids length
                padded = f["labels"] + [-100] * (len(input_ids[i]) - len(f["labels"]))
                labels.append(padded)
            else:
                labels.append(f["labels"])
    
    # Verify lengths match before converting to tensors
    for i in range(len(input_ids)):
        assert len(input_ids[i]) == len(labels[i]), f"Length mismatch: {len(input_ids[i])} vs {len(labels[i])}"
    
    # Convert to tensors with proper padding
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(i, dtype=torch.long) for i in input_ids],
        batch_first=True, 
        padding_value=pad_token
    )
    
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(m, dtype=torch.long) for m in attention_mask],
        batch_first=True, 
        padding_value=0
    )
    
    labels = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(l, dtype=torch.long) for l in labels],
        batch_first=True, 
        padding_value=-100
    )
    
    # Final shape verification
    assert input_ids.shape == labels.shape, f"Shape mismatch: {input_ids.shape} vs {labels.shape}"
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# Initialize wandb
wandb.init(project=project_name, name=run_name)

# Get gradient accumulation steps from config or use default
gradient_accumulation_steps = config.get("gradient_accumulation_steps", 8)
logger.info(f"Using gradient accumulation steps: {gradient_accumulation_steps}")

# Training arguments with memory optimizations
training_args = TrainingArguments(
    overwrite_output_dir=True,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    logging_steps=1,
    bf16=True,
    output_dir=f"./{base_repo_id}",
    report_to="wandb",
    save_steps=save_steps,
    remove_unused_columns=True,
    learning_rate=learning_rate,
    # Memory optimizations
    gradient_checkpointing=True,
    optim="adamw_torch",  # Use 8-bit optimizer to save memory
    max_grad_norm=None,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.01,
    dataloader_num_workers=64,
    ddp_find_unused_parameters=False,
)

# Initialize trainer with the data collator from pretrain/train.py
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    data_collator=data_collator,
)

# Start training
trainer.train()
