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


def load_mtbench_questions(path: str, max_questions: Optional[int] = None) -> List[Dict[str, Any]]:
    questions: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                questions.append(json.loads(line))
            if max_questions is not None and len(questions) >= max_questions:
                break
    return questions


def auto_load_mtbench_questions(max_questions: Optional[int] = None) -> List[Dict[str, Any]]:
    """Try to locate MT-Bench questions from FastChat install or download from GitHub."""
    # 1) Try to locate inside installed fastchat package
    try:
        import fastchat  # type: ignore
        fastchat_dir = Path(fastchat.__file__).parent
        candidates = [
            fastchat_dir / "llm_judge" / "mt_bench" / "question.jsonl",
            fastchat_dir / "llm_judge" / "data" / "mt_bench" / "question.jsonl",
            fastchat_dir.parent / "fastchat" / "llm_judge" / "mt_bench" / "question.jsonl",
        ]
        for cand in candidates:
            if cand.exists():
                return load_mtbench_questions(str(cand), max_questions=max_questions)
    except Exception:
        pass

    # 2) Fallback: try downloading from known URLs (GitHub + Hugging Face)
    try:
        import requests  # type: ignore
        candidate_urls = [
            # FastChat older path
            "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/mt_bench/question.jsonl",
            # FastChat alternative path with data subdir
            "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/mt_bench/data/question.jsonl",
            # FastChat alternative data layout
            "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl",
            # Hugging Face Space (preferred stable link)
            "https://huggingface.co/spaces/lmsys/mt-bench/resolve/main/data/mt_bench/question.jsonl",
        ]
        last_error: Optional[Exception] = None
        for url in candidate_urls:
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                text = resp.text
                # Some hosts may return HTML; do a simple sanity check
                if "</html>" in text.lower() and "{" not in text:
                    continue
                lines = text.strip().splitlines()
                questions: List[Dict[str, Any]] = []
                for line in lines:
                    if line.strip():
                        questions.append(json.loads(line))
                    if max_questions is not None and len(questions) >= max_questions:
                        break
                if questions:
                    return questions
            except Exception as url_err:
                last_error = url_err
                continue
        raise last_error or FileNotFoundError("All candidate URLs failed")
    except Exception as e:
        raise FileNotFoundError(
            "Could not find MT-Bench question.jsonl locally and failed to download from known mirrors. "
            "Please provide --question_file pointing to a valid question.jsonl."
        ) from e


def build_chat_messages(turns: List[str], answers_so_far: List[str]) -> str:
    history_parts: List[str] = []
    for i, user_turn in enumerate(turns[: len(answers_so_far) + 1]):
        history_parts.append("### Instruction:")
        history_parts.append(user_turn.strip())
        history_parts.append("")
        history_parts.append("### Response:")
        if i < len(answers_so_far):
            history_parts.append(answers_so_far[i].strip())
            history_parts.append("")
    return "\n".join(history_parts).strip() + "\n"


@torch.inference_mode()
def answer_mtbench(
    model,
    tokenizer,
    questions: List[Dict[str, Any]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    for q in questions:
        qid = q.get("question_id") or q.get("id")
        turns: List[str] = q["turns"]
        answers: List[str] = []
        for _ in range(len(turns)):
            prompt_text = build_chat_messages(turns, answers)
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
            answers.append(text.strip())

        outputs.append({
            "question_id": qid,
            "choices": [
                {
                    "index": 0,
                    "turns": answers,
                }
            ],
        })
    return outputs


def save_mtbench_answers(path: str, records: List[Dict[str, Any]]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main(
    base_model: str,
    adapter_dir: Optional[str],
    question_file: Optional[str],
    output_dir: str,
    run_base: bool,
    max_questions: Optional[int],
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
    if question_file:
        questions = load_mtbench_questions(question_file, max_questions=max_questions)
    else:
        print("No --question_file provided. Loading MT-Bench questions automatically...")
        questions = auto_load_mtbench_questions(max_questions=max_questions)

    # Fine-tuned model answers
    if adapter_dir is not None:
        base = build_base_model(base_model, load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit, attn_impl=attn_impl)
        model = PeftModel.from_pretrained(base, adapter_dir)
        model.eval()
        answers_ft = answer_mtbench(model, tokenizer, questions, max_new_tokens, temperature, top_p)
        save_mtbench_answers(str(Path(output_dir) / "mtbench_finetuned_answers.jsonl"), answers_ft)

    # Base model answers
    if run_base:
        base_only = build_base_model(base_model, load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit, attn_impl=attn_impl)
        base_only.eval()
        answers_base = answer_mtbench(base_only, tokenizer, questions, max_new_tokens, temperature, top_p)
        save_mtbench_answers(str(Path(output_dir) / "mtbench_base_answers.jsonl"), answers_base)

    print("Done. Answers saved in:")
    if adapter_dir is not None:
        print(str(Path(output_dir) / "mtbench_finetuned_answers.jsonl"))
    if run_base:
        print(str(Path(output_dir) / "mtbench_base_answers.jsonl"))
    print(
        "Next: run FastChat judging, e.g.:\n"
        "  python -m fastchat.llm_judge.gen_judgment \\\n+           --model-list gpt-4o-mini \\\n+           --answer-file outputs/mtbench_finetuned_answers.jsonl,outputs/mtbench_base_answers.jsonl \\\n+           --ref-answer-file mt_bench/reference_answers.jsonl \\\n+           --judge-file outputs/mtbench_judgments.jsonl\n"
        "Then summarize with:\n"
        "  python -m fastchat.llm_judge.show_result --judge-file outputs/mtbench_judgments.jsonl"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate MT-Bench answers for local models.")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter_dir", type=str, default=None, help="Path to LoRA adapter dir. If omitted, only base is run.")
    parser.add_argument("--question_file", type=str, default=None, help="Path to MT-Bench questions JSONL. If not provided, questions will be auto-loaded from FastChat or downloaded.")
    parser.add_argument("--output_dir", type=str, default="outputs/eval_mtbench")
    parser.add_argument("--run_base", action="store_true")
    parser.add_argument("--max_questions", type=int, default=None)
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
        question_file=args.question_file,
        output_dir=args.output_dir,
        run_base=args.run_base,
        max_questions=args.max_questions,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        attn_impl=args.attn_impl,
    )

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


def run_mt_bench(
    merged_model_dir: str,
    output_json: str = "outputs/mtbench_answers.json",
    judge_model: str = "gpt-4o-mini",
    num_questions: int = 40,
):
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
        json.dump({
            "note": "Use FastChat MT-Bench CLI to generate and judge.",
            "suggested": {
                "judge_model": judge_model,
                "num_questions": num_questions,
                "commands": [
                    "# Example (after serving the merged model):",
                    "python -m fastchat.eval.mt_bench --model-path http://localhost:8000 --num-questions %d --judge-model %s" % (num_questions, judge_model),
                ],
            },
        }, f, indent=2)
    print("MT-Bench: placeholder written to", output_json)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare and run MT-Bench using FastChat")
    parser.add_argument("--model_dir", type=str, default="outputs/llama2-7b-dolly-lora")
    parser.add_argument("--merged_out", type=str, default="outputs/merged-for-fastchat")
    parser.add_argument("--answers_json", type=str, default="outputs/mtbench_answers.json")
    parser.add_argument("--judge_model", type=str, default="gpt-4o-mini", help="OpenAI judge model for MT-Bench")
    parser.add_argument("--num_questions", type=int, default=40, help="Limit the number of MT-Bench questions")
    args = parser.parse_args()

    merged_path = prepare_fastchat_model(args.model_dir, args.merged_out)
    print("Merged model saved to:", merged_path)
    run_mt_bench(merged_path, args.answers_json, judge_model=args.judge_model, num_questions=args.num_questions)


