import json
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def build_base_model(model_name: str, load_in_4bit: bool, load_in_8bit: bool, attn_impl: Optional[str] = None):
    device_map = "auto"
    if load_in_4bit or load_in_8bit:
        try:
            import bitsandbytes as bnb  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Requested 4/8-bit loading but bitsandbytes is not available."
            ) from e

    if load_in_4bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device_map,
            load_in_4bit=True,
        )
    elif load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device_map,
            load_in_8bit=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=device_map,
        )

    if attn_impl is not None:
        try:
            model.config.attn_implementation = attn_impl
        except Exception:
            pass

    return model


def load_prompts_jsonl(path: str, max_examples: Optional[int] = None) -> List[Dict[str, Any]]:
    prompts: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                prompts.append(json.loads(line))
            if max_examples is not None and len(prompts) >= max_examples:
                break
    return prompts


def format_prompt(example: Dict[str, Any]) -> str:
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
    prompts: List[Dict[str, Any]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[str]:
    outputs: List[str] = []
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
    out_path: str,
    prompts: List[Dict[str, Any]],
    generations: List[str],
    generator_name: str,
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
    base_model: str,
    adapter_dir: Optional[str],
    prompts_file: str,
    output_dir: str,
    run_base: bool,
    max_examples: Optional[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    load_in_4bit: bool,
    load_in_8bit: bool,
    attn_impl: Optional[str],
):
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    tokenizer = load_tokenizer(base_model)
    prompts = load_prompts_jsonl(prompts_file, max_examples=max_examples)

    # Fine-tuned (LoRA) model
    if adapter_dir is not None:
        base = build_base_model(base_model, load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit, attn_impl=attn_impl)
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
        base_only = build_base_model(base_model, load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit, attn_impl=attn_impl)
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
    parser.add_argument("--prompts_file", type=str, required=True, help="Path to AlpacaEval 2 prompts JSONL.")
    parser.add_argument("--output_dir", type=str, default="outputs/eval_alpacaeval2")
    parser.add_argument("--run_base", action="store_true", help="Also generate outputs for the base model.")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--attn_impl", type=str, default=None)

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
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        attn_impl=args.attn_impl,
    )

import json
import os
import re
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

    # Save generated outputs in alpaca-eval expected format (DataFrame-like JSON or list of dicts)
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrame format that alpaca-eval expects
    import pandas as pd
    df_outputs = pd.DataFrame(outputs)
    # Save as JSON lines (one dict per line) which is more compatible
    output_jsonl = str(output_json).replace(".json", ".jsonl")
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    # Also save as JSON for compatibility
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)
    print(f"Generated responses saved to: {output_json} ({len(outputs)} examples)")

    # Now run the judge evaluation using Python API
    print(f"Running judge evaluation with {judge_model}...")
    import subprocess
    import sys
    
    try:
        # Use alpaca_eval Python API with correct format
        print("Using Python API for evaluation...")
        from alpaca_eval import evaluate as alpaca_evaluate
        from alpaca_eval import utils as alpaca_utils
        
        # Load model outputs - alpaca-eval expects DataFrame with 'output' column
        # Ensure we have the right columns
        model_df = pd.DataFrame(outputs)
        if "output" not in model_df.columns:
            raise ValueError("Model outputs must have 'output' column")
        
        # Get reference outputs (baseline) - load from alpaca_eval baseline
        reference_df = None
        try:
            # Load baseline - this will download if needed
            reference_df = alpaca_utils.load_or_convert_to_dataframe("alpaca_eval_gpt4_baseline")
            
            # Ensure reference has 'output' column
            if "output" not in reference_df.columns:
                print("Warning: Reference baseline does not have 'output' column, skipping reference comparison")
                reference_df = None
            else:
                # Match the number of examples
                if len(model_df) < len(reference_df):
                    reference_df = reference_df.head(len(model_df)).copy()
                elif len(model_df) > len(reference_df):
                    # Truncate model outputs to match reference
                    model_df = model_df.head(len(reference_df)).copy()
        except Exception as e:
            print(f"Warning: Could not load baseline reference: {e}")
            print("Will evaluate without reference baseline (comparison will be limited)")
            reference_df = None
        
        # Run evaluation - use file path (most reliable method)
        print("Running evaluation (this may take a few minutes)...")
        
        # Use the JSONL file we created earlier (or create if needed)
        model_output_file = str(output_json).replace(".json", ".jsonl")
        if not Path(model_output_file).exists():
            # Create JSONL if it doesn't exist
            with open(model_output_file, "w", encoding="utf-8") as f:
                for item in outputs:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        # Try evaluate with file path (most compatible)
        try:
            results_dict = alpaca_evaluate(
                model_outputs=model_output_file,
                reference_outputs=reference_df,  # Can be None or DataFrame
                annotators_config=judge_model,
                is_avoid_reannotations=True,
                max_workers=1,
            )
        except Exception as e:
            # If that fails, try with DataFrame
            print(f"File path method failed ({e}), trying DataFrame...")
            results_dict = alpaca_evaluate(
                model_outputs=model_df,
                reference_outputs=reference_df,
                annotators_config=judge_model,
                is_avoid_reannotations=True,
                max_workers=1,
            )
        
        # Extract win rate from results
        win_rate = results_dict.get('win_rate', None)
        if win_rate is None:
            # Try to extract from other fields
            if 'df_leaderboard' in results_dict:
                df_lb = results_dict['df_leaderboard']
                if len(df_lb) > 0:
                    win_rate = df_lb.iloc[0].get('win_rate', None)
        
        n_total = results_dict.get('n_total', len(model_df))
        
        # Format as stdout-like string
        result_output = f"Win Rate: {win_rate:.1%}\n" if win_rate is not None else "Win Rate: N/A\n"
        result_output += f"Total Evaluations: {n_total}\n"
        result_output += f"\nDetailed Results:\n{json.dumps({k: v for k, v in results_dict.items() if k != 'df_leaderboard'}, indent=2, default=str)}"
        
        class MockResult:
            returncode = 0
            stdout = result_output
            stderr = ""
        
        result = MockResult()
        
    except Exception as api_error:
        # Fallback: try to use CLI or provide manual instructions
        print(f"Python API failed: {api_error}")
        print(f"\nError details: {str(api_error)}")
        
        # Try CLI if available
        import shutil
        alpaca_eval_cmd = shutil.which("alpaca-eval")
        
        if alpaca_eval_cmd:
            print("Trying CLI as fallback...")
            cmd = [
                alpaca_eval_cmd,
                "evaluate",
                "--model_outputs", str(output_json),
                "--annotators_config", judge_model,
                "--is_avoid_reannotations",
                "--max_workers", "1",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                print(f"CLI also failed: {result.stderr}")
                raise RuntimeError(f"Both Python API and CLI failed. API error: {api_error}")
        else:
            raise RuntimeError(f"Python API failed and CLI not found. Error: {api_error}\nYou can manually run: alpaca-eval evaluate --model_outputs {output_json} --annotators_config {judge_model}")
        
        if hasattr(result, 'returncode') and result.returncode == 0 or (not hasattr(result, 'returncode') and result.stdout):
            print("\n" + "="*70)
            print("AlpacaEval 2 Results")
            print("="*70)
            print(result.stdout)
            
            # Parse and display key metrics
            lines = result.stdout.split("\n")
            win_rate = None
            n_total = None
            
            for line in lines:
                # Try to extract win_rate (format varies: "win_rate: 0.XX" or "XX%")
                if "win_rate" in line.lower() or "win rate" in line.lower():
                    # Extract numbers from the line
                    numbers = re.findall(r'[\d.]+', line)
                    if numbers:
                        try:
                            win_rate = float(numbers[0])
                            if win_rate > 1.0:  # If it's a percentage (e.g., 75.5)
                                win_rate = win_rate / 100.0
                        except:
                            pass
                if "n_total" in line.lower() or "total" in line.lower():
                    numbers = re.findall(r'\d+', line)
                    if numbers:
                        n_total = int(numbers[-1])
            
            # Display comparison summary
            print("\n" + "-"*70)
            print("📊 Performance Summary")
            print("-"*70)
            
            if win_rate is not None:
                baseline_rate = 0.50  # AlpacaEval baseline is typically 50% (ties)
                improvement = win_rate - baseline_rate
                improvement_pct = improvement * 100
                
                print(f"✅ Win Rate: {win_rate:.1%}")
                print(f"📈 Baseline Win Rate: {baseline_rate:.1%}")
                
                if improvement > 0:
                    print(f"🎉 Improvement: +{improvement_pct:.1f} percentage points")
                    print(f"💪 Your fine-tuned model performs better than baseline!")
                elif improvement < 0:
                    print(f"⚠️  Change: {improvement_pct:.1f} percentage points")
                    print(f"💡 Model may need more training or hyperparameter tuning")
                else:
                    print(f"➡️  Performance similar to baseline")
                
                if n_total:
                    print(f"\n📋 Evaluated on {n_total} examples")
            else:
                print("⚠️  Could not parse win_rate from results. Check full output above.")
            
            print("="*70)
            
            # Save results summary
            results_summary_path = str(Path(output_json).parent / f"{Path(output_json).stem}_summary.txt")
            with open(results_summary_path, "w", encoding="utf-8") as f:
                f.write(result.stdout)
                if win_rate is not None:
                    f.write(f"\n\nSummary:\n")
                    f.write(f"Win Rate: {win_rate:.1%}\n")
                    f.write(f"Baseline: {baseline_rate:.1%}\n")
                    f.write(f"Improvement: {improvement_pct:+.1f}pp\n")
            print(f"\n✅ Full results saved to: {results_summary_path}")
            print(f"✅ Model outputs saved to: {output_json}")
            
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


