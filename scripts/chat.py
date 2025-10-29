from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = "You are a helpful assistant."


def load_merged_model(adapter_dir: str):
    from peft import PeftModel

    cfg = (Path(adapter_dir) / "adapter_config.json").read_text(encoding="utf-8")
    import json as _json
    base_model = _json.loads(cfg)["base_model"]

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    return tokenizer, model


def chat_once(tokenizer, model, prompt: str, max_new_tokens: int = 512) -> str:
    full_prompt = f"{SYSTEM_PROMPT}\n\n### Instruction:\n{prompt}\n\n### Response:\n"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text.split("### Response:")[-1].strip()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chat with the fine-tuned LoRA model")
    parser.add_argument("--adapter_dir", type=str, default="outputs/llama2-7b-dolly-lora")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    tok, mod = load_merged_model(args.adapter_dir)
    reply = chat_once(tok, mod, args.prompt, args.max_new_tokens)
    print("Response:\n", reply)


