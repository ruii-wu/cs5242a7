# Distributed Training with FSDP (Fully Sharded Data Parallel)
# For bonus marks: Multi-GPU training with PyTorch FSDP

import os
import torch
from dataclasses import dataclass, field
from typing import List
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    HfArgumentParser
)
from peft import LoraConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# ============================================================================
# DISTRIBUTED TRAINING CONFIGURATION
# ============================================================================

@dataclass
class DistributedTrainingArguments(TrainingArguments):
    """Extended training arguments for distributed training"""
    
    # Model
    model_name: str = field(default="meta-llama/Llama-2-7b-hf")
    
    # LoRA
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    
    # Dataset
    dataset_name: str = field(default="databricks/databricks-dolly-15k")
    max_seq_length: int = field(default=512)
    
    # Training
    output_dir: str = field(default="./llama2-dolly-fsdp")
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=4)
    learning_rate: float = field(default=2e-4)
    warmup_steps: int = field(default=100)
    
    # Distributed settings
    fsdp: str = field(
        default="full_shard auto_wrap",
        metadata={"help": "FSDP configuration"}
    )
    fsdp_config: dict = field(
        default_factory=lambda: {
            "fsdp_transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"],
            "fsdp_backward_prefetch": "backward_pre",
            "fsdp_cpu_ram_efficient_loading": True,
        }
    )
    
    # Optimization
    bf16: bool = field(default=True)
    tf32: bool = field(default=True)
    gradient_checkpointing: bool = field(default=True)
    optim: str = field(default="adamw_torch")
    lr_scheduler_type: str = field(default="cosine")
    
    # Logging
    logging_steps: int = field(default=10)
    eval_steps: int = field(default=100)
    save_steps: int = field(default=500)
    evaluation_strategy: str = field(default="steps")
    save_strategy: str = field(default="steps")
    load_best_model_at_end: bool = field(default=True)
    report_to: str = field(default="tensorboard")
    
    # Memory optimization
    max_grad_norm: float = field(default=1.0)
    save_total_limit: int = field(default=3)

# ============================================================================
# DATASET PREPARATION
# ============================================================================

def format_dolly_dataset(example):
    """Format Dolly dataset into instruction format"""
    instruction = example.get("instruction", "").strip()
    context = example.get("context", "").strip()
    response = example.get("response", "").strip()
    
    if not response:
        return {"text": ""}
    
    if context:
        text = f"""### Instruction:
{instruction}

### Context:
{context}

### Response:
{response}"""
    else:
        text = f"""### Instruction:
{instruction}

### Response:
{response}"""
    
    return {"text": text}

def prepare_dataset(dataset_name: str):
    """Load and prepare dataset"""
    dataset = load_dataset(dataset_name, split="train")
    
    # Format
    dataset = dataset.map(
        format_dolly_dataset,
        remove_columns=dataset.column_names,
        num_proc=4
    )
    
    # Filter empty
    dataset = dataset.filter(lambda x: len(x["text"]) > 0)
    
    # Split
    train_val_test = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_val_test["train"]
    
    val_test = train_val_test["test"].train_test_split(test_size=0.5, seed=42)
    val_dataset = val_test["train"]
    test_dataset = val_test["test"]
    
    return train_dataset, val_dataset, test_dataset

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_distributed():
    """Main distributed training function"""
    
    # Parse arguments
    parser = HfArgumentParser(DistributedTrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    # Print configuration
    if args.local_rank in [-1, 0]:
        print("="*80)
        print("DISTRIBUTED TRAINING CONFIGURATION")
        print("="*80)
        print(f"Model: {args.model_name}")
        print(f"Dataset: {args.dataset_name}")
        print(f"Output: {args.output_dir}")
        print(f"Distributed Strategy: FSDP")
        print(f"World Size: {torch.cuda.device_count()} GPUs")
        print(f"Batch Size per Device: {args.per_device_train_batch_size}")
        print(f"Gradient Accumulation: {args.gradient_accumulation_steps}")
        print(f"Effective Batch Size: {args.per_device_train_batch_size * args.gradient_accumulation_steps * torch.cuda.device_count()}")
        print("="*80)
    
    # Prepare dataset
    train_dataset, val_dataset, test_dataset = prepare_dataset(args.dataset_name)
    
    if args.local_rank in [-1, 0]:
        print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    
    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        use_cache=False,  # Required for gradient checkpointing
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # LoRA configuration
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Data collator
    response_template = "### Response:"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer
    )
    
    # Initialize trainer with FSDP
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        data_collator=collator,
        packing=False,
    )
    
    # Train
    if args.local_rank in [-1, 0]:
        print("\nStarting distributed training with FSDP...")
    
    trainer.train()
    
    # Save model (only on main process)
    if args.local_rank in [-1, 0]:
        print("\nSaving model...")
        trainer.save_model(args.output_dir)
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_metrics = trainer.evaluate(eval_dataset=test_dataset)
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)
        
        print("\n" + "="*80)
        print("Distributed training complete!")
        print(f"Model saved to: {args.output_dir}")
        print("="*80)

# ============================================================================
# RUN SCRIPT
# ============================================================================

if __name__ == "__main__":
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Distributed training requires GPUs.")
        exit(1)
    
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPU(s)")
    
    if num_gpus < 2:
        print("Warning: Only 1 GPU detected. FSDP benefits are minimal.")
        print("Consider using the standard training script instead.")
    
    train_distributed()


# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================

"""
To run distributed training:

1. Single node, multiple GPUs (FSDP):
   accelerate launch --config_file fsdp_config.yaml train_fsdp.py

2. Or with torchrun (PyTorch native):
   torchrun --nproc_per_node=4 train_fsdp.py \\
       --output_dir ./llama2-dolly-fsdp \\
       --num_train_epochs 3 \\
       --per_device_train_batch_size 4

3. Multiple nodes (advanced):
   On each node:
   torchrun --nproc_per_node=4 \\
       --nnodes=2 \\
       --node_rank=$NODE_RANK \\
       --master_addr=$MASTER_ADDR \\
       --master_port=$MASTER_PORT \\
       train_fsdp.py

FSDP Benefits:
- Shards model parameters across GPUs
- Reduces memory per GPU
- Enables training larger models
- Better scaling efficiency than DDP for large models
"""