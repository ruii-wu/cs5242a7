"""
MT-Bench evaluation script with baseline comparison.

This script evaluates your fine-tuned model and compares it against a baseline model
to show improvement metrics.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_for_eval(model_path: str):
    """Load a model for evaluation (handles both merged and base models)."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    return tokenizer, model


def run_mtbench_evaluation(
    model_path_or_endpoint: str,
    output_dir: str,
    judge_model: str = "gpt-4o-mini",
    num_questions: int = 40,
    is_endpoint: bool = False,
) -> Dict:
    """
    Run MT-Bench evaluation using FastChat.
    
    Args:
        model_path_or_endpoint: Model path or API endpoint URL
        output_dir: Directory to save results
        judge_model: Judge model name
        num_questions: Number of questions to evaluate
        is_endpoint: Whether model_path_or_endpoint is an API endpoint
    
    Returns:
        Dictionary with evaluation results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if is_endpoint:
        model_path_arg = model_path_or_endpoint
    else:
        model_path_arg = model_path_or_endpoint
    
    # Run FastChat MT-Bench
    cmd = [
        sys.executable, "-m", "fastchat.eval.mt_bench",
        "--model-path", model_path_arg,
        "--judge-model", judge_model,
        "--num-questions", str(num_questions),
        "--save-dir", str(output_dir),
    ]
    
    print(f"Running MT-Bench evaluation...")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    
    if result.returncode != 0:
        print(f"Error running MT-Bench: {result.stderr}")
        return None
    
    # Parse results from saved files
    results_file = output_dir / "results.json"
    if results_file.exists():
        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        return results
    else:
        # Try to parse from stdout
        print("Warning: Could not find results.json, parsing from stdout")
        return {"raw_output": result.stdout}


def calculate_average_score(results: Dict) -> Optional[float]:
    """Extract average score from MT-Bench results."""
    if not results:
        return None
    
    # MT-Bench results structure varies, try common fields
    if "average" in results:
        return float(results["average"])
    elif "overall_score" in results:
        return float(results["overall_score"])
    elif "mean" in results:
        return float(results["mean"])
    elif "scores" in results:
        scores = results["scores"]
        if isinstance(scores, list):
            return sum(float(s) for s in scores) / len(scores) if scores else None
        elif isinstance(scores, dict):
            all_scores = []
            for v in scores.values():
                if isinstance(v, (int, float)):
                    all_scores.append(float(v))
                elif isinstance(v, list):
                    all_scores.extend([float(x) for x in v])
            return sum(all_scores) / len(all_scores) if all_scores else None
    
    return None


def compare_results(
    baseline_results: Dict,
    finetuned_results: Dict,
    baseline_name: str = "Baseline (Llama-2-7B)",
    finetuned_name: str = "Fine-tuned Model",
) -> Dict:
    """Compare baseline and fine-tuned model results."""
    baseline_score = calculate_average_score(baseline_results)
    finetuned_score = calculate_average_score(finetuned_results)
    
    if baseline_score is None or finetuned_score is None:
        return {
            "baseline_score": baseline_score,
            "finetuned_score": finetuned_score,
            "improvement": None,
            "improvement_pct": None,
            "error": "Could not extract scores from results",
        }
    
    improvement = finetuned_score - baseline_score
    improvement_pct = (improvement / baseline_score * 100) if baseline_score > 0 else None
    
    return {
        "baseline_score": baseline_score,
        "finetuned_score": finetuned_score,
        "improvement": improvement,
        "improvement_pct": improvement_pct,
        "baseline_name": baseline_name,
        "finetuned_name": finetuned_name,
    }


def main(
    finetuned_model_dir: str,
    baseline_model: str = "meta-llama/Llama-2-7b-chat-hf",
    output_base_dir: str = "outputs/mtbench_comparison",
    judge_model: str = "gpt-4o-mini",
    num_questions: int = 40,
    use_endpoints: bool = False,
    baseline_endpoint: Optional[str] = None,
    finetuned_endpoint: Optional[str] = None,
):
    """
    Main function to compare baseline and fine-tuned models on MT-Bench.
    
    Args:
        finetuned_model_dir: Path to fine-tuned model (LoRA adapter or merged model)
        baseline_model: Baseline model name or path
        output_base_dir: Base directory for outputs
        judge_model: Judge model for evaluation
        num_questions: Number of MT-Bench questions
        use_endpoints: Whether to use API endpoints instead of local models
        baseline_endpoint: Baseline model API endpoint (if use_endpoints=True)
        finetuned_endpoint: Fine-tuned model API endpoint (if use_endpoints=True)
    """
    output_base_dir = Path(output_base_dir)
    
    print("="*70)
    print("MT-Bench Evaluation with Baseline Comparison")
    print("="*70)
    
    # Evaluate baseline model
    print("\n📊 Step 1: Evaluating Baseline Model...")
    print("-"*70)
    if use_endpoints and baseline_endpoint:
        baseline_results = run_mtbench_evaluation(
            baseline_endpoint,
            output_base_dir / "baseline",
            judge_model=judge_model,
            num_questions=num_questions,
            is_endpoint=True,
        )
    else:
        baseline_results = run_mtbench_evaluation(
            baseline_model,
            output_base_dir / "baseline",
            judge_model=judge_model,
            num_questions=num_questions,
            is_endpoint=False,
        )
    
    if not baseline_results:
        print("❌ Baseline evaluation failed!")
        return
    
    baseline_score = calculate_average_score(baseline_results)
    print(f"✅ Baseline Average Score: {baseline_score:.2f}" if baseline_score else "⚠️  Could not calculate baseline score")
    
    # Evaluate fine-tuned model
    print("\n📊 Step 2: Evaluating Fine-tuned Model...")
    print("-"*70)
    
    # Check if we need to merge LoRA first
    finetuned_model_path = finetuned_model_dir
    if (Path(finetuned_model_dir) / "adapter_config.json").exists():
        print("Merging LoRA adapter for evaluation...")
        # Import prepare_fastchat_model locally
        from peft import PeftModel
        
        cfg_path = Path(finetuned_model_dir) / "adapter_config.json"
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        base_model = cfg["base_model"]
        
        tok = AutoTokenizer.from_pretrained(base_model, use_fast=False)
        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        model = PeftModel.from_pretrained(base, finetuned_model_dir)
        merged = model.merge_and_unload()
        
        merged_dir = output_base_dir / "merged_finetuned"
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(merged_dir)
        tok.save_pretrained(merged_dir)
        finetuned_model_path = str(merged_dir)
        print(f"Merged model saved to: {finetuned_model_path}")
    
    if use_endpoints and finetuned_endpoint:
        finetuned_results = run_mtbench_evaluation(
            finetuned_endpoint,
            output_base_dir / "finetuned",
            judge_model=judge_model,
            num_questions=num_questions,
            is_endpoint=True,
        )
    else:
        finetuned_results = run_mtbench_evaluation(
            finetuned_model_path,
            output_base_dir / "finetuned",
            judge_model=judge_model,
            num_questions=num_questions,
            is_endpoint=False,
        )
    
    if not finetuned_results:
        print("❌ Fine-tuned model evaluation failed!")
        return
    
    finetuned_score = calculate_average_score(finetuned_results)
    print(f"✅ Fine-tuned Average Score: {finetuned_score:.2f}" if finetuned_score else "⚠️  Could not calculate fine-tuned score")
    
    # Compare results
    print("\n" + "="*70)
    print("📈 Comparison Results")
    print("="*70)
    
    comparison = compare_results(baseline_results, finetuned_results)
    
    baseline_score = comparison["baseline_score"]
    finetuned_score = comparison["finetuned_score"]
    improvement = comparison["improvement"]
    improvement_pct = comparison["improvement_pct"]
    
    if baseline_score is not None and finetuned_score is not None:
        print(f"\nBaseline Model ({comparison['baseline_name']}):")
        print(f"  Average Score: {baseline_score:.2f}")
        
        print(f"\nFine-tuned Model ({comparison['finetuned_name']}):")
        print(f"  Average Score: {finetuned_score:.2f}")
        
        if improvement is not None:
            print(f"\n{'='*70}")
            if improvement > 0:
                print(f"🎉 Improvement: +{improvement:.2f} points")
                if improvement_pct:
                    print(f"💪 Relative Improvement: +{improvement_pct:.1f}%")
                print("✅ Your fine-tuned model performs better than baseline!")
            elif improvement < 0:
                print(f"⚠️  Change: {improvement:.2f} points")
                if improvement_pct:
                    print(f"📉 Relative Change: {improvement_pct:.1f}%")
                print("💡 Model may need more training or hyperparameter tuning")
            else:
                print("➡️  Performance similar to baseline")
    
    # Save comparison results
    comparison_file = output_base_dir / "comparison_results.json"
    with open(comparison_file, "w", encoding="utf-8") as f:
        json.dump({
            "comparison": comparison,
            "baseline_full_results": baseline_results,
            "finetuned_full_results": finetuned_results,
        }, f, indent=2)
    
    print(f"\n✅ Full results saved to: {comparison_file}")
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MT-Bench evaluation with baseline comparison")
    parser.add_argument("--finetuned_model_dir", type=str, required=True,
                        help="Path to fine-tuned model (LoRA adapter or merged)")
    parser.add_argument("--baseline_model", type=str, default="meta-llama/Llama-2-7b-chat-hf",
                        help="Baseline model name or path")
    parser.add_argument("--output_base_dir", type=str, default="outputs/mtbench_comparison",
                        help="Base directory for all outputs")
    parser.add_argument("--judge_model", type=str, default="gpt-4o-mini",
                        help="Judge model for evaluation")
    parser.add_argument("--num_questions", type=int, default=40,
                        help="Number of MT-Bench questions to evaluate")
    parser.add_argument("--use_endpoints", action="store_true",
                        help="Use API endpoints instead of local models")
    parser.add_argument("--baseline_endpoint", type=str, default=None,
                        help="Baseline model API endpoint")
    parser.add_argument("--finetuned_endpoint", type=str, default=None,
                        help="Fine-tuned model API endpoint")
    
    args = parser.parse_args()
    
    main(
        finetuned_model_dir=args.finetuned_model_dir,
        baseline_model=args.baseline_model,
        output_base_dir=args.output_base_dir,
        judge_model=args.judge_model,
        num_questions=args.num_questions,
        use_endpoints=args.use_endpoints,
        baseline_endpoint=args.baseline_endpoint,
        finetuned_endpoint=args.finetuned_endpoint,
    )
