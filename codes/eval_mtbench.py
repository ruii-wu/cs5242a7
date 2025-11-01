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
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16
    )
    
    return model

def load_mtbench_questions(path, max_questions=None):
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                questions.append(json.loads(line))
            if max_questions is not None and len(questions) >= max_questions:
                break
    return questions


def auto_load_mtbench_questions(max_questions=None):
    """Try to locate MT-Bench questions from FastChat install or download from GitHub."""
    # 1) Try to locate inside installed fastchat package
    try:
        import fastchat  
        fastchat_dir = Path(fastchat.__file__).parent
        candidates = [
            fastchat_dir / "llm_judge" / "mt_bench" / "question.jsonl",
            fastchat_dir / "llm_judge" / "data" / "mt_bench" / "question.jsonl",
            fastchat_dir.parent / "fastchat" / "llm_judge" / "mt_bench" / "question.jsonl",
            Path("data") / "mt_bench" / "question.jsonl",
        ]
        for cand in candidates:
            if cand.exists():
                print(f"Loading questions from: {str(cand)}")
                return load_mtbench_questions(str(cand), max_questions=max_questions)
    except Exception:
        pass

    # try downloading from known URLs (GitHub + Hugging Face)
    try:
        import requests  # type: ignore
        candidate_urls = [
            # Hugging Face Space (preferred stable link)
            "https://huggingface.co/spaces/lmsys/mt-bench/resolve/main/data/mt_bench/question.jsonl",
            # FastChat main path
            "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl",
        ]
        last_error = None
        for url in candidate_urls:
            try:
                print(f"Downloading questions from: {url}")
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                text = resp.text
                if "</html>" in text.lower() and "{" not in text:
                    continue
                lines = text.strip().splitlines()
                questions = []
                for line in lines:
                    if line.strip():
                        questions.append(json.loads(line))
                    if max_questions is not None and len(questions) >= max_questions:
                        break
                if questions:
                    local_path = Path("data") / "mt_bench" / "question.jsonl"
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(local_path, "w", encoding="utf-8") as f:
                        f.write(text)
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


def build_chat_messages(turns, answers_so_far):
    history_parts = []
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
    questions,
    max_new_tokens,
    temperature,
    top_p,
):
    outputs = []
    for q in questions:
        qid = q.get("question_id") or q.get("id")
        turns = q["turns"]
        answers = []
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


def save_mtbench_answers(path, records):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(
    base_model,
    adapter_dir,
    question_file,
    output_dir,
    base_model_label,
    ft_model_label,
    run_base,
    max_questions,
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
    if question_file:
        questions = load_mtbench_questions(question_file, max_questions=max_questions)
    else:
        print("No --question_file provided. Loading MT-Bench questions automatically...")
        questions = auto_load_mtbench_questions(max_questions=max_questions)

    model_list_parts = []
    base_label = None
    ft_label = None
    
    if run_base:
        base_label = base_model_label or Path(base_model).name
        base_path = Path(output_dir) / f"{base_label}.jsonl"
        model_list_parts.append(base_label)
        
    if adapter_dir is not None:
        ft_label = ft_model_label or Path(adapter_dir).parent.name
        ft_path = Path(output_dir) / f"{ft_label}.jsonl"
        model_list_parts.append(ft_label)

    #Generate Answers
    if adapter_dir is not None:
        print(f"Generating answers for fine-tuned model: {ft_label}...")
        base = build_base_model(base_model)
        model = PeftModel.from_pretrained(base, adapter_dir)
        model.eval()
        answers_ft = answer_mtbench(model, tokenizer, questions, max_new_tokens, temperature, top_p)
        save_mtbench_answers(str(ft_path), answers_ft)
        del model
        del base
        torch.cuda.empty_cache()

    if run_base:
        print(f"Generating answers for base model: {base_label}...")
        base_only = build_base_model(base_model)
        base_only.eval()
        answers_base = answer_mtbench(base_only, tokenizer, questions, max_new_tokens, temperature, top_p)
        save_mtbench_answers(str(base_path), answers_base)
        del base_only
        torch.cuda.empty_cache()
    
    model_list_str = " ".join(model_list_parts)
    max_q_str = str(max_questions) if max_questions else str(len(questions)) # Use actual len

    print("\n" + "="*50)
    print("Done. Answers saved in:")
    if adapter_dir is not None:
        print(f"  {ft_path}")
    if run_base:
        print(f"  {base_path}")

    print("\nNext: run FastChat judging (from project root):\n")
    print(
        f"  python -m fastchat.llm_judge.gen_judgment \\\n"
        f"      --judge-model gpt-4o-mini \\\n"
        f"      --mode pairwise-all \\\n"
        f"      --model-list {model_list_str} \\\n"
        f"      --parallel 4 \\\n"
        f"      --first-n {max_q_str}\n"
    )
    print("Then summarize with:\n")
    print(
        f"  python -m fastchat.llm_judge.show_result \\\n"
        f"      --judge-model gpt-4o-mini \\\n"
        f"      --mode pairwise-all \\\n"
        f"      --model-list {model_list_str}"
    )
    print("="*50 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate MT-Bench answers for local models.")
    parser.add_argument("--base_model", type=str, required=True, help="Path to base model (e.g., meta-llama/Llama-2-7b-hf)")
    parser.add_argument("--adapter_dir", type=str, default=None, help="Path to LoRA adapter checkpoint dir.")
    parser.add_argument("--base_model_label", type=str, default=None, help="Label for base model, e.g., 'llama-2-7b-hf'. Inferred from --base_model if not set.")
    parser.add_argument("--ft_model_label", type=str, default=None, help="Label for adapter model, e.g., 'llama2-7b-dolly-lora'. Inferred from --adapter_dir's parent if not set.")
    parser.add_argument("--question_file", type=str, default=None, help="Path to MT-Bench questions JSONL. If not provided, questions will be auto-loaded.")
    parser.add_argument("--output_dir", type=str, default="data/mt_bench/model_answer", help="Directory to save .jsonl answer files.")
    parser.add_argument("--run_base", action="store_true", help="Also run evaluation on the base model.")
    parser.add_argument("--max_questions", type=int, default=None, help="Limit to first N questions.")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    args = parser.parse_args()
    
    if args.adapter_dir is None and not args.run_base:
        parser.error("You must specify --adapter_dir and/or --run_base")

    main(
        base_model=args.base_model,
        adapter_dir=args.adapter_dir,
        question_file=args.question_file,
        output_dir=args.output_dir,
        base_model_label=args.base_model_label,
        ft_model_label=args.ft_model_label,
        run_base=args.run_base,
        max_questions=args.max_questions,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )