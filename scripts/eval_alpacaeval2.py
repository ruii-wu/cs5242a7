import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def build_base_model(model_name):
    device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16, 
        device_map=device_map,
    )

    return model


def load_prompts_from_alpacaeval(max_examples=None):

    """Load prompts from alpaca-eval package."""
    try:
        from alpaca_eval import utils
        import pandas as pd
        
        # Try to load from alpaca-eval
        try:
            df = utils.load_or_convert_to_dataframe("alpaca_eval_gpt4_baseline")
        except Exception:
            # Fallback: load from Hugging Face dataset
            from datasets import load_dataset
            ds = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", revision="main")
            df = pd.DataFrame(ds["eval"])
        
        # Convert to list of dicts
        prompts = []
        for _, row in df.iterrows():
            prompt_dict = {
                "instruction": row.get("instruction", ""),
                "input": row.get("input", ""),
            }
            prompts.append(prompt_dict)
            if max_examples is not None and len(prompts) >= max_examples:
                break
        
        print(f"Loaded {len(prompts)} prompts from alpaca-eval package")
        return prompts
    except ImportError:
        raise ImportError(
            "alpaca-eval package is required. Install with: pip install alpaca-eval\n"
            "Or provide --prompts_file with a JSONL file containing prompts."
        )


def load_prompts_jsonl(path, max_examples=None):
    """Load prompts from a JSONL file."""
    prompts = [] 
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                prompts.append(json.loads(line))
            if max_examples is not None and len(prompts) >= max_examples:
                break
    print(f"Loaded {len(prompts)} prompts from {path}")
    return prompts


def format_prompt(example):
    instr = example.get("instruction") or example.get("prompt") or example.get("question") or ""
    ctx = example.get("context") or example.get("input") or ""
    parts = [
        "### Instruction:",
        str(instr).strip(),
        "",
        "### Context:",
        str(ctx).strip(),
        "",
        "### Response:",
    ]
    return "\n".join(parts).strip() + "\n"


@torch.inference_mode()
def generate_outputs(
    model,
    tokenizer,
    prompts,
    max_new_tokens,
    temperature,
    top_p,
):

    outputs = [] 
    for ex in prompts:
        prompt_text = format_prompt(ex)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        outputs.append(text.strip())
    return outputs


def save_alpacaeval_outputs(
    out_path,
    prompts,
    generations,
    generator_name,
):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex, out in zip(prompts, generations):
            record = {
                "instruction": ex.get("instruction") or ex.get("prompt") or ex.get("question") or "",
                "output": out,
                "generator": generator_name,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(
    base_model,
    adapter_dir,
    prompts_file,
    output_dir,
    run_base,
    max_examples,
    max_new_tokens,
    temperature,
    top_p,
):
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    tokenizer = load_tokenizer(base_model)
    
    # Load prompts: from file if provided, otherwise from alpaca-eval package
    if prompts_file:
        prompts = load_prompts_jsonl(prompts_file, max_examples=max_examples)
    else:
        print("No --prompts_file provided. Loading prompts from alpaca-eval package...")
        prompts = load_prompts_from_alpacaeval(max_examples=max_examples)

    # Fine-tuned (LoRA) model
    if adapter_dir is not None:
        base = build_base_model(base_model)
        model = PeftModel.from_pretrained(base, adapter_dir)
        model.eval()
        gens = generate_outputs(model, tokenizer, prompts, max_new_tokens, temperature, top_p)
        save_alpacaeval_outputs(
            str(Path(output_dir) / "alpacaeval2_finetuned_outputs.jsonl"),
            prompts,
            gens,
            generator_name="finetuned",
        )

    # Base model
    if run_base:
        base_only = build_base_model(base_model)
        base_only.eval()
        gens_base = generate_outputs(base_only, tokenizer, prompts, max_new_tokens, temperature, top_p)
        save_alpacaeval_outputs(
            str(Path(output_dir) / "alpacaeval2_base_outputs.jsonl"),
            prompts,
            gens_base,
            generator_name="base",
        )

    print("Done. Outputs saved in:")
    if adapter_dir is not None:
        print(str(Path(output_dir) / "alpacaeval2_finetuned_outputs.jsonl"))
    if run_base:
        print(str(Path(output_dir) / "alpacaeval2_base_outputs.jsonl"))
    print(
        "Next: run AlpacaEval 2 CLI to judge, e.g.:\n"
        "  alpaca_eval evaluate --model_outputs outputs/alpacaeval2_finetuned_outputs.jsonl --reference_outputs alpaca_eval/reference/alpaca_eval_gpt4.jsonl --annotators_config gpt-4o-mini\n"
        "Or compare two models:\n"
        "  alpaca_eval evaluate --model_outputs outputs/alpacaeval2_finetuned_outputs.jsonl,outputs/alpacaeval2_base_outputs.jsonl --annotators_config gpt-4o-mini"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate AlpacaEval 2 model outputs for local models.")
    parser.add_argument("--base_model", type=str, required=True, help="e.g., meta-llama/Llama-2-7b-hf")
    parser.add_argument("--adapter_dir", type=str, default=None, help="Path to LoRA adapter directory (fine-tuned outputs). If omitted, only base is run.")
    parser.add_argument("--prompts_file", type=str, default=None, help="Path to AlpacaEval 2 prompts JSONL. If not provided, prompts will be loaded from alpaca-eval package.")
    parser.add_argument("--output_dir", type=str, default="outputs/eval_alpacaeval2")
    parser.add_argument("--run_base", action="store_true", help="Also generate outputs for the base model.")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)

    args = parser.parse_args()
    main(
        base_model=args.base_model,
        adapter_dir=args.adapter_dir,
        prompts_file=args.prompts_file,
        output_dir=args.output_dir,
        run_base=args.run_base,
        max_examples=args.max_examples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
