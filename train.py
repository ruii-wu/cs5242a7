# Complete LLaMA-2-7B Fine-Tuning with LoRA on Dolly-15K
# Simplified version using Hugging Face SFTTrainer

import os
import torch
from dataclasses import dataclass, field
from typing import List, Optional
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    HfArgumentParser
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer
try:
    from trl import DataCollatorForCompletionOnlyLM
except ImportError:
    # For older versions of trl, use the transformer's default collator
    from transformers import DataCollatorForLanguageModeling
    DataCollatorForCompletionOnlyLM = None
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

@dataclass
class ModelArguments:
    model_name: str = field(default="meta-llama/Llama-2-7b-hf")
    use_4bit: bool = field(default=True)
    use_8bit: bool = field(default=False)

@dataclass
class LoraArguments:
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

@dataclass
class DataArguments:
    dataset_name: str = field(default="databricks/databricks-dolly-15k")
    max_seq_length: int = field(default=512)
    train_split_ratio: float = field(default=0.8)
    val_split_ratio: float = field(default=0.1)

@dataclass
class CustomTrainingArguments(TrainingArguments):
    output_dir: str = field(default="./llama2-dolly-lora")
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=4)
    learning_rate: float = field(default=2e-4)
    warmup_steps: int = field(default=100)
    logging_steps: int = field(default=10)
    eval_steps: int = field(default=100)
    save_steps: int = field(default=500)
    bf16: bool = field(default=True)
    optim: str = field(default="paged_adamw_32bit")
    lr_scheduler_type: str = field(default="cosine")
    save_total_limit: int = field(default=3)
    gradient_checkpointing: bool = field(default=True)
    evaluation_strategy: str = field(default="steps")
    save_strategy: str = field(default="steps")
    # --- FIX ---
    # Set to False to bypass the check. The override for evaluation_strategy
    # is not being picked up, defaulting to "no", which conflicts with
    # save_strategy="steps" when load_best_model_at_end=True.
    load_best_model_at_end: bool = field(default=False)
    report_to: str = field(default="tensorboard")

# ============================================================================
# 2. DATASET PREPARATION
# ============================================================================

def format_dolly_dataset(example):
    """Format Dolly dataset into instruction format"""
    instruction = example.get("instruction", "").strip()
    context = example.get("context", "").strip()
    response = example.get("response", "").strip()
    
    # Skip empty responses
    if not response:
        return {"text": ""}
    
    # Build formatted text
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

def prepare_dataset(dataset_name: str, train_ratio: float, val_ratio: float):
    """Load and prepare the dataset"""
    print(f"Loading dataset: {dataset_name}")
    
    # Load dataset
    dataset = load_dataset(dataset_name, split="train")
    
    # Format dataset
    print("Formatting dataset...")
    dataset = dataset.map(
        format_dolly_dataset,
        remove_columns=dataset.column_names,
        num_proc=4
    )
    
    # Filter out empty entries
    dataset = dataset.filter(lambda x: len(x["text"]) > 0)
    
    print(f"Total examples after filtering: {len(dataset)}")
    
    # Create splits
    test_ratio = 1.0 - train_ratio - val_ratio
    
    # First split: train vs (val + test)
    train_val_test = dataset.train_test_split(test_size=1-train_ratio, seed=42)
    train_dataset = train_val_test["train"]
    
    # Second split: val vs test
    val_test = train_val_test["test"].train_test_split(
        test_size=test_ratio/(val_ratio + test_ratio),
        seed=42
    )
    val_dataset = val_test["train"]
    test_dataset = val_test["test"]
    
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset

# ============================================================================
# 3. MODEL SETUP
# ============================================================================

def create_model_and_tokenizer(model_args: ModelArguments):
    """Initialize model and tokenizer"""
    
    # Quantization config
    if model_args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif model_args.use_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        bnb_config = None
    
    # Load model
    print(f"Loading model: {model_args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        use_cache=False,  # Required for gradient checkpointing
    )
    
    # Prepare for k-bit training
    if model_args.use_4bit or model_args.use_8bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True
        )
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Load tokenizer
    print(f"Loading tokenizer: {model_args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    return model, tokenizer

# ============================================================================
# 4. TRAINING
# ============================================================================

def train():
    """Main training function"""
    
    # Parse arguments
    parser = HfArgumentParser((
        ModelArguments,
        LoraArguments,
        DataArguments,
        CustomTrainingArguments
    ))
    model_args, lora_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Set up output directory
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    print("="*80)
    print("Configuration:")
    print(f"  Model: {model_args.model_name}")
    print(f"  Dataset: {data_args.dataset_name}")
    print(f"  Output: {training_args.output_dir}")
    print(f"  LoRA r={lora_args.lora_r}, alpha={lora_args.lora_alpha}")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print("="*80)
    
    # Prepare dataset
    train_dataset, val_dataset, test_dataset = prepare_dataset(
        data_args.dataset_name,
        data_args.train_split_ratio,
        data_args.val_split_ratio
    )
    
    # Create model and tokenizer
    model, tokenizer = create_model_and_tokenizer(model_args)
    
    # LoRA configuration
    peft_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        target_modules=lora_args.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Create data collator for completion only (only train on responses)
    response_template = "### Response:"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer
    )
    
    # Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=data_args.max_seq_length,
        tokenizer=tokenizer,
        data_collator=collator,
        packing=False,  # Don't pack multiple examples
    )
    
    # Train
    print("\nStarting training...")
    train_result = trainer.train()
    
    # Save model
    print("\nSaving model...")
    trainer.save_model(training_args.output_dir)
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    # Plot training curves
    plot_training_curves(trainer, training_args.output_dir)
    
    # Evaluate on test set
    # Note: This will run, but evaluation strategy is likely "no"
    # so metrics might be limited.
    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset)
    trainer.log_metrics("test", test_metrics)
    trainer.save_metrics("test", test_metrics)
    
    print("\n" + "="*80)
    print("Training complete!")
    print(f"Model saved to: {training_args.output_dir}")
    print("="*80)
    
    return trainer

# ============================================================================
# 5. VISUALIZATION
# ============================================================================

def plot_training_curves(trainer, output_dir):
    """Plot training and validation loss"""
    log_history = trainer.state.log_history
    
    # Extract losses
    train_logs = [x for x in log_history if "loss" in x and "eval_loss" not in x]
    eval_logs = [x for x in log_history if "eval_loss" in x]
    
    if not train_logs:
        print("No training logs found for plotting")
        return
    
    train_steps = [x["step"] for x in train_logs]
    train_loss = [x["loss"] for x in train_logs]
    
    eval_steps = [x["step"] for x in eval_logs] if eval_logs else []
    eval_loss = [x["eval_loss"] for x in eval_logs] if eval_logs else []
    
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training loss
    axes[0].plot(train_steps, train_loss, linewidth=2, color='#2E86AB')
    axes[0].set_xlabel('Training Steps', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_facecolor('#F8F9FA')
    
    # Validation loss
    if eval_steps:
        axes[1].plot(eval_steps, eval_loss, linewidth=2, color='#F77F00')
        axes[1].set_xlabel('Training Steps', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].set_title('Validation Loss', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_facecolor('#F8F9FA')
    else:
        # Hide axis if no eval data
        axes[1].text(0.5, 0.5, 'No Evaluation Data',
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform=axes[1].transAxes,
                     fontsize=12, color='gray')
        axes[1].set_title('Validation Loss', fontsize=14, fontweight='bold')
        axes[1].set_facecolor('#F8F9FA')


    
    plt.tight_layout()
    
    # Save
    save_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining curves saved to: {save_path}")
    plt.close()

# ============================================================================
# 6. INFERENCE
# ============================================================================

def test_inference(model_path: str = "./llama2-dolly-lora"):
    """Test the fine-tuned model"""
    from peft import AutoPeftModelForCausalLM
    
    print(f"\nLoading fine-tuned model from {model_path}...")
    
    # Load model and tokenizer
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Test prompts
    test_prompts = [
        "Explain quantum computing in simple terms.",
        "Write a Python function to reverse a string.",
        "What are the benefits of regular exercise?",
    ]
    
    print("\n" + "="*80)
    print("Testing model generation:")
    print("="*80)
    
    for prompt in test_prompts:
        formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:"
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
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
        
        # Extract only the response
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()
        
        print(f"\nInstruction: {prompt}")
        print(f"Response: {response}")
        print("-"*80)

# ============================================================================
# 7. MAIN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Check if we should run training or inference
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Test inference
        model_path = sys.argv[2] if len(sys.argv) > 2 else "./llama2-dolly-lora"
        test_inference(model_path)
    else:
        # Run training
        trainer = train()
        
        # Test the model
        print("\n" + "="*80)
        print("Testing trained model...")
        print("="*80)
        test_inference(trainer.args.output_dir)
