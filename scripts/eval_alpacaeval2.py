import json
import os
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def build_pipeline(model_dir: str, base_model: str = None, device: str = None):
    from peft import PeftModel

    if device is None:
        device = 0 if torch.cuda.is_available() else -1

    # Load base model and merge LoRA if available
    base = base_model or json.load(open(Path(model_dir) / "adapter_config.json", "r", encoding="utf-8")).get("base_model")
    tok = AutoTokenizer.from_pretrained(base, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    base_m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    try:
        merged = PeftModel.from_pretrained(base_m, model_dir)
        merged = merged.merge_and_unload()
    except Exception:
        merged = base_m

    gen = pipeline(
        "text-generation",
        model=merged,
        tokenizer=tok,
        device=device,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    return gen


def run_alpacaeval(
    model_dir: str,
    output_json: str = "outputs/alpacaeval2_answers.json",
    judge_model: str = "gpt-4o-mini",
    max_examples: int = 300,
):
    # Check for OpenAI API key
    if os.environ.get("OPENAI_API_KEY") is None:
        raise ValueError("OPENAI_API_KEY environment variable must be set for judging. Please set it before running evaluation.")

    # Import alpaca_eval modules
    try:
        from alpaca_eval import utils
        import pandas as pd
    except ImportError:
        raise ImportError("alpaca-eval not properly installed. Try: pip install alpaca-eval[openai]")

    gen = build_pipeline(model_dir)

    # Load the alpaca eval prompts (subset if needed)
    try:
        current_prompts = utils.load_or_convert_to_dataframe("alpaca_eval_gpt4_baseline")
    except Exception:
        # Fallback: try loading from default dataset
        from datasets import load_dataset
        ds = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", revision="main")
        current_prompts = pd.DataFrame(ds["eval"])

    if max_examples > 0 and len(current_prompts) > max_examples:
        print(f"Limiting evaluation to {max_examples} examples (from {len(current_prompts)}) to manage costs.")
        current_prompts = current_prompts.head(max_examples)

    print(f"Generating responses for {len(current_prompts)} prompts...")
    
    # Generate responses using our model
    outputs = []
    for idx, row in current_prompts.iterrows():
        # Extract instruction field (handles both dict-like and Series)
        if isinstance(row, pd.Series):
            instruction = row.get("instruction", "")
        else:
            instruction = row.get("instruction", "") if hasattr(row, "get") else (row["instruction"] if "instruction" in row else "")
        
        if not instruction or not instruction.strip():
            continue
        
        # Format prompt similar to Dolly format for consistency
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        try:
            result = gen(prompt, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.95, pad_token_id=gen.tokenizer.eos_token_id)
            generated_text = result[0]["generated_text"]
            # Extract only the response part
            if "### Response:" in generated_text:
                response = generated_text.split("### Response:")[-1].strip()
            else:
                response = generated_text.replace(prompt, "").strip()
        except Exception as e:
            print(f"Error generating response for prompt {idx}: {e}")
            response = ""
        
        outputs.append({"instruction": instruction, "output": response})
        
        if (len(outputs)) % 10 == 0:
            print(f"Processed {len(outputs)}/{len(current_prompts)} prompts...")

    # Save generated outputs in alpaca-eval expected format (list of dicts with "output" field)
    # Format: [{"instruction": "...", "output": "..."}, ...]
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)
    print(f"Generated responses saved to: {output_json} ({len(outputs)} examples)")

    # Now run the judge evaluation using command-line interface
    print(f"Running judge evaluation with {judge_model}...")
    import subprocess
    import sys
    
    try:
        # Use alpaca-eval CLI for judging
        cmd = [
            sys.executable, "-m", "alpaca_eval",
            "evaluate",
            "--model_outputs", str(output_json),
            "--annotators_config", judge_model,
            "--is_avoid_reannotations",
            "--max_workers", "1",
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        if result.returncode == 0:
            print("\n=== AlpacaEval 2 Results ===")
            print(result.stdout)
            print(f"\nFull results saved to: {output_json}")
            return result.stdout
        else:
            print(f"Judging encountered errors: {result.stderr}")
            print(f"Generated responses are saved to {output_json}")
            print("You can manually run: alpaca-eval evaluate --model_outputs <file> --annotators_config gpt-4o-mini")
            return None
            
    except Exception as e:
        print(f"Error during judging: {e}")
        print(f"Responses saved to {output_json}; you can manually evaluate them later.")
        print(f"Try running: alpaca-eval evaluate --model_outputs {output_json} --annotators_config {judge_model}")
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run AlpacaEval 2 with a HF LoRA model")
    parser.add_argument("--model_dir", type=str, default="outputs/llama2-7b-dolly-lora")
    parser.add_argument("--output_json", type=str, default="outputs/alpacaeval2_answers.json")
    parser.add_argument("--judge_model", type=str, default="gpt-4o-mini", help="OpenAI judge model for AlpacaEval 2")
    parser.add_argument("--max_examples", type=int, default=300, help="Limit the number of eval prompts to control cost")
    args = parser.parse_args()
    Path(Path(args.output_json).parent).mkdir(parents=True, exist_ok=True)
    run_alpacaeval(args.model_dir, args.output_json, judge_model=args.judge_model, max_examples=args.max_examples)


