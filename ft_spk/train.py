from datasets import load_dataset
from datasets import load_from_disk
from datasets import concatenate_datasets
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
import unsloth
from unsloth import FastLanguageModel
# from pretrain.train import data_collator
# from pretrain.train import data_collator
from trl import SFTTrainer
import torch
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

config_file = "config.yaml"


with open(config_file, "r") as file:
    config = yaml.safe_load(file)

dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
dsn = config["TTS_datasets"]
model_name = config["model_name"]
run_name = config["run_name"]
project_name = config["project_name"]
base_repo_id = config["save_folder"]
epochs = config["epochs"]
batch_size = config["batch_size"]
save_steps = config["save_steps"]
pad_token = config["pad_token"]
number_processes = config["number_processes"]
learning_rate = float(config["learning_rate"])
print(f"Using learning rate: {learning_rate}")

# Define a maximum sequence length to prevent OOM errors
# You can adjust this based on your GPU memory
max_length = 8192

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="flash_attention_2")

# Replace the single dataset loading with multiple dataset loading
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

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    full_finetuning = True,
)
model = FastLanguageModel.get_peft_model(
    model,
    r = 0,  # Disable LoRA (full finetuning)
    use_gradient_checkpointing = "unsloth",  # Optimized checkpointing
)

# ds = load_from_disk(dsn)
# ds = load_dataset(dsn, split="train") 

# Updated custom data collator with sequence length truncation
def data_collator(features):
    """
    Efficient data collator that handles padding and truncation uniformly.
    Based on the pretrain implementation for better batch handling.
    """
    input_ids = [f["input_ids"] for f in features]
    
    # Handle missing attention_mask
    if any("attention_mask" not in f for f in features):
        attention_mask = [[1]*len(ids) for ids in input_ids]
    else:
        attention_mask = [f["attention_mask"] for f in features]
    
    # Handle missing labels
    if any("labels" not in f for f in features):
        labels = input_ids
    else:
        labels = [f["labels"] for f in features]
    
    # Pad sequences efficiently
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
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

wandb.init(project=project_name, name=run_name)

# Update training arguments to include gradient accumulation and memory optimizations
# Set environment variable for memory management
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

# max_length = config.get("max_sequence_length", max_length)

# Update training arguments
training_args = TrainingArguments(
    overwrite_output_dir=True,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
    logging_steps=1,
    bf16=True,
    output_dir=f"./{base_repo_id}",
    report_to="wandb",
    save_steps=save_steps,
    remove_unused_columns=True,
    learning_rate=learning_rate,
    # Add learning rate scheduler
    lr_scheduler_type="linear",  # Linear decay of learning rate
    warmup_ratio=0.001,        
    weight_decay = 0.01,
    # Memory optimizations
    fp16_full_eval=True,
    dataloader_num_workers=1,
    optim="adamw_8bit",
    max_grad_norm=1.0,
    gradient_checkpointing=True,  # Add gradient checkpointing
    ddp_find_unused_parameters=False,
)


# training_args = TrainingArguments(
#     num_train_epochs = config["epochs"],
#     per_device_train_batch_size = config["batch_size"],
#     gradient_accumulation_steps = config.get("gradient_accumulation_steps", 8),
#     learning_rate = config["learning_rate"],
#     logging_steps = 1,
#     output_dir = config["save_folder"],
#     optim = "paged_adamw_8bit",  # Better for 8-bit
#     bf16 = True,  # Use with H100
#     fsdp = "full_shard auto_wrap",
#     fsdp_config = {
#         "forward_prefetch": True,
#         "use_orig_params": True,
#     },
#     gradient_checkpointing = True,
#     report_to = "wandb",
#     save_steps = config["save_steps"],
#     ddp_find_unused_parameters = False,
#     dataloader_num_workers = 1,
# )


# Use Unsloth's optimized SFTTrainer
trainer = SFTTrainer(
    model = model,
    train_dataset = ds,
    max_seq_length = max_length,
    tokenizer = tokenizer,
    args = training_args,
    data_collator=data_collator,  # Add this line
    packing = False,  # Disable packing since we're using custom collator
)

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=ds,
#     data_collator=custom_data_collator,
# )

trainer.train()