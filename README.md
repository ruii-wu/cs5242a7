## LLaMA-2-7B LoRA on Dolly-15K (PEFT)

### Setup
- Python 3.10+ recommended
- Install in two steps to avoid indexing issues:

```bash
# 1) Install PyTorch from cu124 index (CUDA 12.x compatible)
pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124

# 2) Install the remaining dependencies from PyPI
pip install -r requirements.txt
```

**Note:** You need Hugging Face authentication to download LLaMA-2-7B:
```bash
huggingface-cli login
# Follow prompts to enter your HF token
```

### 1) Prepare Dataset
```bash
python scripts/prep_dolly.py --output_dir data/dolly15k_prepared
```

### 2) Train (LoRA)
```bash
python scripts/train_lora.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --data_dir data/dolly15k_prepared \
    --output_dir outputs/llama2-7b-dolly-lora \
    --epochs 3 --batch_size 2 --grad_accum_steps 8 \
    --max_length 1024 --lr 2e-4 \
    --dataloader_num_workers 8 --disable_gradient_checkpointing
```

### 3) Plot Loss Curves
```bash
python scripts/plot_losses.py \
  --trainer_state outputs/llama2-7b-dolly-lora/checkpoint-2250/trainer_state.json \
  --out_png outputs/loss_curve_final.png
```

### 4) Evaluation (AlpacaEval 2 + MT-Bench)

Generate outputs for both the fine-tuned adapter and the base model, then use official evaluators to score and compare.

AlpacaEval 2 outputs:
```bash
# Automatically loads prompts from alpaca-eval package (recommended - no --prompts_file needed)
python scripts/eval_alpacaeval2.py \
  --base_model meta-llama/Llama-2-7b-hf \
  --adapter_dir outputs/llama2-7b-dolly-lora/checkpoint-2250 \
  --output_dir outputs/eval_alpacaeval2 \
  --run_base \
  --max_examples 300 \
  --max_new_tokens 512 --temperature 0.2 --top_p 0.95

# Or provide your own prompts file (optional):
# python scripts/eval_alpacaeval2.py ... --prompts_file path/to/alpacaeval2_prompts.jsonl ...

# Judge and compare (pick an annotator within budget)
alpaca_eval evaluate \
  --model_outputs outputs/eval_alpacaeval2/alpacaeval2_finetuned_outputs.jsonl,outputs/eval_alpacaeval2/alpacaeval2_base_outputs.jsonl \
  --annotators_config gpt-4o-mini
```

MT-Bench answers:
```bash
python scripts/eval_mtbench.py \
  --base_model meta-llama/Llama-2-7b-hf \
  --adapter_dir outputs/llama2-7b-dolly-lora/checkpoint-2250 \
  --output_dir outputs/eval_mtbench \
  --run_base \
  --max_questions 40 \
  --max_new_tokens 512 --temperature 0.2 --top_p 0.95

# Judge with FastChat and summarize
python -m fastchat.llm_judge.gen_judgment \
  --model-list gpt-4o-mini \
  --answer-file outputs/eval_mtbench/mtbench_finetuned_answers.jsonl,outputs/eval_mtbench/mtbench_base_answers.jsonl \
  --ref-answer-file path/to/mt_bench/reference_answers.jsonl \
  --judge-file outputs/eval_mtbench/mtbench_judgments.jsonl

python -m fastchat.llm_judge.show_result \
  --judge-file outputs/eval_mtbench/mtbench_judgments.jsonl
```

Notes:
- AlpacaEval 2 prompts are automatically loaded from the `alpaca-eval` package if `--prompts_file` is not provided.
- Use FastChat's official `question.jsonl` and `reference_answers.jsonl` for MT-Bench.
- You can reduce cost with `--max_examples` / `--max_questions` and cheaper annotators.

### Notes
- The `requirements.txt` pins versions for reproducibility.
- Use `gpt-4o-mini` (or other mini models) for judging to stay within budget.
- Keep evaluation subset sizes (`--max_examples`, `--num_questions`) conservative to fit ~$10 budget.
- PyTorch cu124 wheels are self-contained; a system CUDA 12.8 install is fine.
- For multi-GPU or distributed training, integrate Accelerate/DeepSpeed or FSDP as needed.


