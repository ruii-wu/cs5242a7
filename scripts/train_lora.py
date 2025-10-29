import json
import math
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model

def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def build_model(model_name): 
    device_map = "auto"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
    )
    return model


def get_lora_model(base_model, r, alpha, dropout, target_modules=None):
    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    lora_model = get_peft_model(base_model, lora_cfg)
    lora_model.print_trainable_parameters()
    return lora_model


def load_jsonl_dataset(path):
    ds = load_dataset("json", data_files={
        "train": f"{path}/train.jsonl",
        "validation": f"{path}/validation.jsonl",
        "test": f"{path}/test.jsonl",
    })
    return ds


def tokenize_function(examples, tokenizer, max_length):
    return tokenizer(examples["text"], truncation=True, max_length=max_length, padding=False)


def main(
    model_name="meta-llama/Llama-2-7b-hf",
    data_dir="data/dolly15k_prepared",
    output_dir="outputs/llama2-7b-dolly-lora",
    max_length=1024,
    lr=2e-4,
    epochs=3,
    batch_size=2,
    grad_accum_steps=8,
    warmup_ratio=0.03,
    weight_decay=0.0,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    fp16=True,
    bf16=False,
    disable_gradient_checkpointing=False,
    dataloader_num_workers=4,
):
    
    # Enable TF32 on Ampere+ to speed up matmul while keeping good accuracy
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    tokenizer = load_tokenizer(model_name)
    model = build_model(model_name) 
    model = get_lora_model(
        model,
        r=lora_r,
        alpha=lora_alpha,
        dropout=lora_dropout,
        target_modules=None,
    )

    ds = load_jsonl_dataset(data_dir)
    tokenized = ds.map(lambda x: tokenize_function(x, tokenizer, max_length=max_length), batched=True, remove_columns=["text"]) 

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    steps_per_epoch = math.ceil(len(tokenized["train"]) / (batch_size * max(1, torch.cuda.device_count()) * grad_accum_steps))
    save_steps = max(50, steps_per_epoch)
    eval_steps = max(100, min(steps_per_epoch // 5, 200))

    # Prefer bf16 when supported; otherwise fall back to fp16 if requested
    bf16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    use_bf16 = bool(bf16 and bf16_supported)
    use_fp16 = bool(fp16 and not use_bf16)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=eval_steps,
        logging_steps=10,
        save_steps=save_steps,
        save_total_limit=2,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        dataloader_num_workers=dataloader_num_workers,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        lr_scheduler_type="cosine",
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_checkpointing=not disable_gradient_checkpointing,
        logging_first_step=True,
        report_to=["none"],
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)

    # Save final adapter config path for downstream eval
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "adapter_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "base_model": model_name,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
        }, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LoRA fine-tuning for LLaMA-2-7B on Dolly-15K")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--data_dir", type=str, default="data/dolly15k_prepared")
    parser.add_argument("--output_dir", type=str, default="outputs/llama2-7b-dolly-lora")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=16)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--no_fp16", action="store_true")
    parser.add_argument("--no_bf16", action="store_true")
    parser.add_argument("--disable_gradient_checkpointing", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    args = parser.parse_args()

    main(
        model_name=args.model_name,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_length=args.max_length,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        fp16=not args.no_fp16,
        bf16=not args.no_bf16,
        disable_gradient_checkpointing=args.disable_gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
    )