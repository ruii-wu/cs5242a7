import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def prepare_fastchat_model(model_dir: str, merged_out: str = "outputs/merged-for-fastchat") -> str:
    """
    Merge LoRA adapter into base weights for easier serving with FastChat.
    Returns the path containing merged weights.
    """
    from peft import PeftModel

    cfg_path = Path(model_dir) / "adapter_config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    base_model = cfg["base_model"]

    tok = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    model = PeftModel.from_pretrained(base, model_dir)
    merged = model.merge_and_unload()

    out_dir = Path(merged_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return str(out_dir)


def run_mt_bench(merged_model_dir: str, output_json: str = "outputs/mtbench_answers.json"):
    """
    This script assumes you will use FastChat's MT-Bench harness.
    Steps (manual commands):
      1) Serve the model:  python -m fastchat.serve.vllm_worker --model <merged_model_dir>
      2) Launch controller and gradio/web servers (FastChat docs)
      3) Run:  python -m fastchat.eval.mt_bench --model-path <endpoint> ...
    Here we only keep a placeholder to indicate data paths and write where results go.
    """
    Path(Path(output_json).parent).mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"note": "Use FastChat MT-Bench CLI to generate and judge."}, f)
    print("MT-Bench: placeholder written to", output_json)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare and run MT-Bench using FastChat")
    parser.add_argument("--model_dir", type=str, default="outputs/llama2-7b-dolly-lora")
    parser.add_argument("--merged_out", type=str, default="outputs/merged-for-fastchat")
    parser.add_argument("--answers_json", type=str, default="outputs/mtbench_answers.json")
    args = parser.parse_args()

    merged_path = prepare_fastchat_model(args.model_dir, args.merged_out)
    print("Merged model saved to:", merged_path)
    run_mt_bench(merged_path, args.answers_json)


