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
    # Lazy import to avoid hard dep when not evaluating
    from alpaca_eval.main import main as alpaca_main

    gen = build_pipeline(model_dir)

    # AlpacaEval 2 loads prompts internally; we provide a callable model
    # Provide OpenAI key if using GPT-based judges
    if os.environ.get("OPENAI_API_KEY") is None:
        print("WARNING: OPENAI_API_KEY not set. Judged results may not run.")

    # Create a lightweight wrapper exposing a generate function signature
    class HFModelWrapper:
        def __init__(self, pipe):
            self.pipe = pipe

        def __call__(self, prompts: List[str]) -> List[str]:
            outs = self.pipe(prompts, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.95, pad_token_id=self.pipe.tokenizer.eos_token_id)
            texts = []
            for o in outs:
                # pipeline returns list[dict] per input when batch; normalize
                last = o[0]["generated_text"] if isinstance(o, list) else o["generated_text"]
                texts.append(last)
            return texts

    model_wrapper = HFModelWrapper(gen)

    # Run evaluation (this will download benchmark and run judging if configured)
    # The API supports passing a python callable model
    # Try passing judge model and subset size when supported by the API
    try:
        results = alpaca_main(
            model=model_wrapper,
            output_path=output_json,
            judge_model=judge_model,
            num_examples=max_examples,
        )
    except TypeError:
        # Fallback if current alpaca-eval version doesn't support these kwargs
        print("alpaca_eval.main.main signature does not support judge_model/num_examples; running defaults.")
        results = alpaca_main(model=model_wrapper, output_path=output_json)
    print("AlpacaEval 2 results saved to:", output_json)
    print(results)


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


