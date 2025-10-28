# Evaluation Scripts for AlpacaEval 2 and MT-Bench
# Run after fine-tuning is complete

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
import subprocess

# ============================================================================
# SETUP
# ============================================================================

MODEL_PATH = "./llama2-dolly-lora"  # Path to fine-tuned model
BASE_MODEL = "meta-llama/Llama-2-7b-hf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_finetuned_model(model_path: str):
    """Load the fine-tuned LoRA model"""
    print(f"Loading base model from {BASE_MODEL}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    print(f"Loading LoRA weights from {model_path}...")
    model = PeftModel.from_pretrained(base_model, model_path)
    model = model.merge_and_unload()  # Merge LoRA weights
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    """Generate response for a given prompt"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the prompt from response
    if prompt in response:
        response = response[len(prompt):].strip()
    
    return response

# ============================================================================
# ALPACAEVAL 2 EVALUATION
# ============================================================================

def run_alpacaeval():
    """
    Run AlpacaEval 2 evaluation
    
    Installation:
    pip install alpaca-eval
    
    Usage:
    This function generates responses and runs AlpacaEval
    """
    print("\n" + "="*80)
    print("ALPACAEVAL 2 EVALUATION")
    print("="*80)
    
    # Check if alpaca_eval is installed
    try:
        import alpaca_eval
    except ImportError:
        print("ERROR: alpaca_eval not installed. Install with:")
        print("pip install alpaca-eval")
        return
    
    # Load model
    model, tokenizer = load_finetuned_model(MODEL_PATH)
    
    # Load AlpacaEval dataset
    print("Loading AlpacaEval dataset...")
    from alpaca_eval import get_alpaca_eval_data
    eval_data = get_alpaca_eval_data()
    
    # Generate responses
    print(f"Generating responses for {len(eval_data)} examples...")
    outputs = []
    
    for example in tqdm(eval_data):
        instruction = example["instruction"]
        
        # Format prompt (using the instruction format from training)
        prompt = f"### Instruction:\n{instruction}\n\n### Response:"
        
        response = generate_response(model, tokenizer, prompt, max_new_tokens=512)
        
        outputs.append({
            "instruction": instruction,
            "output": response,
            "generator": "llama2-dolly-lora"
        })
    
    # Save outputs
    output_path = "alpacaeval_outputs.json"
    with open(output_path, "w") as f:
        json.dump(outputs, f, indent=2)
    
    print(f"Outputs saved to {output_path}")
    
    # Run AlpacaEval
    print("\nRunning AlpacaEval evaluation...")
    try:
        from alpaca_eval import evaluate
        results = evaluate(
            model_outputs=outputs,
            annotators_config="alpaca_eval_gpt4",
        )
        
        print("\nAlpacaEval Results:")
        print(json.dumps(results, indent=2))
        
        # Save results
        with open("alpacaeval_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
    except Exception as e:
        print(f"Error running AlpacaEval: {e}")
        print("You can run evaluation manually with:")
        print(f"alpaca_eval --model_outputs {output_path}")

# ============================================================================
# MT-BENCH EVALUATION
# ============================================================================

def run_mtbench():
    """
    Run MT-Bench evaluation using FastChat
    
    Installation:
    pip install fschat
    
    MT-Bench evaluates multi-turn conversation quality
    """
    print("\n" + "="*80)
    print("MT-BENCH EVALUATION")
    print("="*80)
    
    # Check if FastChat is installed
    try:
        import fastchat
    except ImportError:
        print("ERROR: fastchat not installed. Install with:")
        print("pip install fschat")
        return
    
    # Load model
    model, tokenizer = load_finetuned_model(MODEL_PATH)
    
    # Load MT-Bench questions
    print("Loading MT-Bench questions...")
    try:
        # Download MT-Bench questions if not present
        questions_url = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl"
        questions_path = "mt_bench_questions.jsonl"
        
        if not os.path.exists(questions_path):
            print(f"Downloading MT-Bench questions from {questions_url}")
            import urllib.request
            urllib.request.urlretrieve(questions_url, questions_path)
        
        # Load questions
        questions = []
        with open(questions_path, "r") as f:
            for line in f:
                questions.append(json.loads(line))
        
        print(f"Loaded {len(questions)} questions")
        
    except Exception as e:
        print(f"Error loading MT-Bench questions: {e}")
        return
    
    # Generate responses
    print("Generating responses for MT-Bench...")
    answers = []
    
    for question in tqdm(questions):
        question_id = question["question_id"]
        turns = question["turns"]
        category = question["category"]
        
        conversation = []
        for turn_idx, turn in enumerate(turns):
            # Build conversation context
            if turn_idx == 0:
                prompt = f"### Instruction:\n{turn}\n\n### Response:"
            else:
                # Include previous turns
                context = ""
                for i, prev_turn in enumerate(conversation):
                    context += f"User: {turns[i]}\nAssistant: {prev_turn}\n\n"
                prompt = f"{context}User: {turn}\nAssistant:"
            
            response = generate_response(model, tokenizer, prompt, max_new_tokens=512)
            conversation.append(response)
        
        answers.append({
            "question_id": question_id,
            "answer_id": f"llama2-dolly-lora-{question_id}",
            "model_id": "llama2-dolly-lora",
            "choices": [{"index": 0, "turns": conversation}],
            "tstamp": 0
        })
    
    # Save answers
    answers_path = "mt_bench_answers.jsonl"
    with open(answers_path, "w") as f:
        for answer in answers:
            f.write(json.dumps(answer) + "\n")
    
    print(f"Answers saved to {answers_path}")
    
    # Instructions for running MT-Bench judgment
    print("\nTo complete MT-Bench evaluation, run:")
    print("python -m fastchat.llm_judge.gen_judgment \\")
    print(f"    --model-list llama2-dolly-lora \\")
    print(f"    --judge-model gpt-4 \\")
    print(f"    --answer-dir . \\")
    print(f"    --question-file {questions_path}")
    print("\nThen view results with:")
    print("python -m fastchat.llm_judge.show_result --model-list llama2-dolly-lora")

# ============================================================================
# ADDITIONAL BENCHMARKS (BONUS)
# ============================================================================

def run_mmlu():
    """
    Run MMLU (Massive Multitask Language Understanding) evaluation
    
    Installation:
    pip install lm-eval
    """
    print("\n" + "="*80)
    print("MMLU EVALUATION (BONUS)")
    print("="*80)
    
    print("Running MMLU evaluation...")
    print("This requires the lm-evaluation-harness package:")
    print("pip install lm-eval")
    print("\nRun with:")
    print(f"lm_eval --model hf \\")
    print(f"    --model_args pretrained={MODEL_PATH} \\")
    print(f"    --tasks mmlu \\")
    print(f"    --batch_size 8")

def run_gsm8k():
    """Run GSM8K (math reasoning) evaluation"""
    print("\n" + "="*80)
    print("GSM8K EVALUATION (BONUS)")
    print("="*80)
    
    print("Running GSM8K evaluation...")
    print("Run with lm-evaluation-harness:")
    print(f"lm_eval --model hf \\")
    print(f"    --model_args pretrained={MODEL_PATH} \\")
    print(f"    --tasks gsm8k \\")
    print(f"    --batch_size 8")

def run_truthfulqa():
    """Run TruthfulQA evaluation"""
    print("\n" + "="*80)
    print("TRUTHFULQA EVALUATION (BONUS)")
    print("="*80)
    
    print("Running TruthfulQA evaluation...")
    print("Run with lm-evaluation-harness:")
    print(f"lm_eval --model hf \\")
    print(f"    --model_args pretrained={MODEL_PATH} \\")
    print(f"    --tasks truthfulqa \\")
    print(f"    --batch_size 8")

# ============================================================================
# COMPARE BASE VS FINETUNED
# ============================================================================

def compare_models():
    """
    Compare base model vs fine-tuned model on sample instructions
    """
    print("\n" + "="*80)
    print("BASE MODEL VS FINE-TUNED MODEL COMPARISON")
    print("="*80)
    
    # Test instructions
    test_instructions = [
        "Explain the concept of machine learning to a 10-year-old.",
        "Write a Python function to calculate the Fibonacci sequence.",
        "What are the key differences between supervised and unsupervised learning?",
        "Describe the process of photosynthesis in simple terms.",
        "How do neural networks work?"
    ]
    
    # Load models
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_tokenizer.pad_token = base_tokenizer.eos_token
    
    print("Loading fine-tuned model...")
    ft_model, ft_tokenizer = load_finetuned_model(MODEL_PATH)
    
    # Generate comparisons
    comparisons = []
    
    for instruction in test_instructions:
        print(f"\n{'='*80}")
        print(f"Instruction: {instruction}")
        print(f"{'='*80}")
        
        prompt = f"### Instruction:\n{instruction}\n\n### Response:"
        
        # Base model response
        print("\nBase model response:")
        base_response = generate_response(base_model, base_tokenizer, prompt, max_new_tokens=256)
        print(base_response)
        
        # Fine-tuned model response
        print("\nFine-tuned model response:")
        ft_response = generate_response(ft_model, ft_tokenizer, prompt, max_new_tokens=256)
        print(ft_response)
        
        comparisons.append({
            "instruction": instruction,
            "base_response": base_response,
            "finetuned_response": ft_response
        })
    
    # Save comparisons
    with open("model_comparison.json", "w") as f:
        json.dump(comparisons, f, indent=2)
    
    print("\nComparison saved to model_comparison.json")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluation scripts for fine-tuned LLaMA-2")
    parser.add_argument("--alpacaeval", action="store_true", help="Run AlpacaEval 2")
    parser.add_argument("--mtbench", action="store_true", help="Run MT-Bench")
    parser.add_argument("--compare", action="store_true", help="Compare base vs fine-tuned")
    parser.add_argument("--mmlu", action="store_true", help="Run MMLU (bonus)")
    parser.add_argument("--gsm8k", action="store_true", help="Run GSM8K (bonus)")
    parser.add_argument("--truthfulqa", action="store_true", help="Run TruthfulQA (bonus)")
    parser.add_argument("--all", action="store_true", help="Run all evaluations")
    
    args = parser.parse_args()
    
    if args.all or args.alpacaeval:
        run_alpacaeval()
    
    if args.all or args.mtbench:
        run_mtbench()
    
    if args.all or args.compare:
        compare_models()
    
    if args.mmlu:
        run_mmlu()
    
    if args.gsm8k:
        run_gsm8k()
    
    if args.truthfulqa:
        run_truthfulqa()
    
    if not any(vars(args).values()):
        print("No evaluation specified. Use --help for options.")
        print("\nQuick start:")
        print("  python evaluate.py --compare     # Compare base vs fine-tuned")
        print("  python evaluate.py --alpacaeval  # Run AlpacaEval 2")
        print("  python evaluate.py --mtbench     # Run MT-Bench")
        print("  python evaluate.py --all         # Run all main evaluations")