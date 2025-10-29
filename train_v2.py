# Training Script WITHOUT 4-bit Quantization
# Works on Windows, Linux, and any system without bitsandbytes
# Requires more GPU memory (40GB+ recommended) or uses CPU

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
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    # Model
    model_name: str = field(default="meta-llama/Llama-2-7b-hf")
    
    # LoRA
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    
    # Dataset
    dataset_name: str = field(default="databricks/databricks-dolly-15k")
    max_seq_length: int = field(default=512)
    
    # Training
    output_dir: str = field(default="./llama2-dolly-lora")
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=1)  # Reduced for no quantization
    gradient_accumulation_steps: int = field(default=16)  # Increased to compensate
    learning_rate: float = field(default=2e-4)
    warmup_steps: int = field(default=100)
    logging_steps: int = field(default=10)
    eval_steps: int = field(default=100)
    save_steps: int = field(default=500)
    
    # Memory optimization
    gradient_checkpointing: bool = field(default=True)
    use_cpu_offload: bool = field(default=False)  # Set True if GPU memory is limited

# ============================================================================
# DATASET PREPARATION
# ============================================================================

def format_dolly_dataset(example):
    """Format dataset with proper prompt structure"""
    instruction = example.get("instruction", "").strip()
    context = example.get("context", "").strip()
    response = example.get("response", "").strip()
    
    if not response:
        return {"text": ""}
    
    if context:
        text = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{context}

### Response:
{response}"""
    else:
        text = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{response}"""
    
    return {"text": text}

def prepare_dataset(dataset_name: str):
    """Load and prepare the dataset"""
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")
    
    print("Formatting dataset...")
    dataset = dataset.map(
        format_dolly_dataset,
        remove_columns=dataset.column_names,
        num_proc=4
    )
    
    dataset = dataset.filter(lambda x: len(x["text"]) > 0)
    print(f"Total examples: {len(dataset)}")
    
    # Split
    train_val = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_val["train"]
    
    val_test = train_val["test"].train_test_split(test_size=0.5, seed=42)
    val_dataset = val_test["train"]
    test_dataset = val_test["test"]
    
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset

# ============================================================================
# MODEL SETUP (NO QUANTIZATION)
# ============================================================================

def create_model_and_tokenizer(config: Config):
    """Initialize model and tokenizer WITHOUT quantization"""
    
    print(f"Loading model: {config.model_name}")
    print("NOTE: Loading model in FP16/BF16 without quantization")
    print("This requires more GPU memory but avoids bitsandbytes dependency")
    
    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Using dtype: {dtype}")
    else:
        device = "cpu"
        dtype = torch.float32
        print("WARNING: No GPU detected. Training will be VERY slow on CPU.")
        print("Consider using Google Colab or a cloud GPU instance.")
    
    # Load model
    try:
        if config.use_cpu_offload:
            # Load with CPU offloading for limited GPU memory
            print("Loading with CPU offloading (slower but uses less GPU memory)...")
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=dtype,
                device_map="auto",
                offload_folder="offload",
                trust_remote_code=True,
            )
        else:
            # Load entirely to GPU/CPU
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
    except torch.cuda.OutOfMemoryError:
        print("\nERROR: GPU Out of Memory!")
        print("Solutions:")
        print("1. Set use_cpu_offload=True in Config")
        print("2. Reduce batch_size to 1")
        print("3. Use a machine with more GPU memory (40GB+ recommended)")
        print("4. Install bitsandbytes and use 4-bit quantization")
        raise
    
    # Enable gradient checkpointing
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled (saves memory)")
    
    # Load tokenizer
    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return model, tokenizer

# ============================================================================
# TRAINING
# ============================================================================

def train(config=None):
    """Main training function"""
    
    # Parse config
    if config is None:
        parser = HfArgumentParser(Config)
        try:
            import sys
            if 'ipykernel' in sys.modules:
                config = Config()
            else:
                config = parser.parse_args_into_dataclasses()[0]
        except:
            config = Config()
    
    print("="*80)
    print("CONFIGURATION (NO QUANTIZATION)")
    print("="*80)
    print(f"Model: {config.model_name}")
    print(f"Dataset: {config.dataset_name}")
    print(f"Output: {config.output_dir}")
    print(f"LoRA r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"Batch size: {config.per_device_train_batch_size} × {config.gradient_accumulation_steps} = {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    print(f"Gradient checkpointing: {config.gradient_checkpointing}")
    print(f"CPU offload: {config.use_cpu_offload}")
    print("="*80)
    
    # Prepare dataset
    train_dataset, val_dataset, test_dataset = prepare_dataset(config.dataset_name)
    
    # Create model and tokenizer
    model, tokenizer = create_model_and_tokenizer(config)
    
    # Apply LoRA
    print("\nApplying LoRA...")
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        gradient_checkpointing=config.gradient_checkpointing,
        optim="adamw_torch",  # Use standard optimizer (no paged_adamw)
        lr_scheduler_type="cosine",
        report_to="tensorboard",
        save_total_limit=3,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )
    
    # Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        tokenizer=tokenizer,
        packing=False,
    )
    
    # Train
    print("\nStarting training...")
    print("NOTE: Without quantization, training will use more memory.")
    print("Monitor GPU memory with: nvidia-smi")
    print()
    
    try:
        train_result = trainer.train()
    except torch.cuda.OutOfMemoryError:
        print("\nGPU Out of Memory during training!")
        print("Try:")
        print("1. Reduce per_device_train_batch_size to 1")
        print("2. Increase gradient_accumulation_steps")
        print("3. Set use_cpu_offload=True")
        print("4. Use gradient_checkpointing=True")
        raise
    
    # Save model
    print("\nSaving model...")
    trainer.save_model(config.output_dir)
    
    # Save metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset)
    trainer.log_metrics("test", test_metrics)
    trainer.save_metrics("test", test_metrics)
    
    # Plot training curves
    plot_training_curves(trainer, config.output_dir)
    
    print("\n" + "="*80)
    print("Training complete!")
    print(f"Model saved to: {config.output_dir}")
    print("="*80)
    
    return trainer

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_curves(trainer, output_dir):
    """Plot training and validation loss"""
    log_history = trainer.state.log_history
    
    train_logs = [x for x in log_history if "loss" in x and "eval_loss" not in x]
    eval_logs = [x for x in log_history if "eval_loss" in x]
    
    if not train_logs:
        print("No training logs found")
        return
    
    train_steps = [x["step"] for x in train_logs]
    train_loss = [x["loss"] for x in train_logs]
    
    eval_steps = [x["step"] for x in eval_logs] if eval_logs else []
    eval_loss = [x["eval_loss"] for x in eval_logs] if eval_logs else []
    
    fig, axes = plt.subplots(1, 2 if eval_steps else 1, figsize=(14 if eval_steps else 7, 5))
    if not eval_steps:
        axes = [axes]
    
    axes[0].plot(train_steps, train_loss, linewidth=2, color='#2E86AB')
    axes[0].set_xlabel('Steps', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    if eval_steps:
        axes[1].plot(eval_steps, eval_loss, linewidth=2, color='#F77F00')
        axes[1].set_xlabel('Steps', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].set_title('Validation Loss', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {save_path}")
    plt.close()

# ============================================================================
# INFERENCE
# ============================================================================

def test_model(model_path: str = "./llama2-dolly-lora"):
    """Test the fine-tuned model"""
    from peft import AutoPeftModelForCausalLM
    
    print(f"\nLoading model from {model_path}...")
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    test_prompts = [
        "Explain quantum computing in simple terms.",
        "Write a Python function to calculate factorial.",
        "What are the benefits of regular exercise?",
    ]
    
    print("\n" + "="*80)
    print("TESTING MODEL")
    print("="*80)
    
    for prompt in test_prompts:
        full_prompt = f"""Below is an instruction that describes a task.

### Instruction:
{prompt}

### Response:"""
        
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()
        
        print(f"\nInstruction: {prompt}")
        print(f"Response: {response}")
        print("-"*80)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    try:
        get_ipython()
        IN_JUPYTER = True
    except NameError:
        IN_JUPYTER = False
    
    if IN_JUPYTER:
        print("Running in Jupyter Notebook with default configuration")
        trainer = train()
        print("\nTesting trained model...")
        test_model(trainer.args.output_dir)
    elif len(sys.argv) > 1 and sys.argv[1] == "--test":
        model_path = sys.argv[2] if len(sys.argv) > 2 else "./llama2-dolly-lora"
        test_model(model_path)
    else:
        trainer = train()
        print("\nTesting trained model...")
        test_model(trainer.args.output_dir)