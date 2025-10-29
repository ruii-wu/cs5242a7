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
    --no_fp16 --dataloader_num_workers 8 --disable_gradient_checkpointing
```

Optional memory-saving:
- Add `--load_in_8bit` or `--load_in_4bit` (requires bitsandbytes).

### 3) Plot Loss Curves
```bash
python scripts/plot_losses.py \
  --trainer_state outputs/llama2-7b-dolly-lora/trainer_state.json \
  --out_png outputs/loss_curve.png
```

### 4) Evaluate

AlpacaEval 2（建议使用 mini 评测模型以节省开销；需 `OPENAI_API_KEY`）
```bash
# Set OpenAI API key (required for judging)
# Linux/Mac: export OPENAI_API_KEY="sk-..."
# Windows CMD: set OPENAI_API_KEY=sk-...
# Windows PowerShell: $env:OPENAI_API_KEY="sk-..."

python scripts/eval_alpacaeval2.py \
  --model_dir outputs/llama2-7b-dolly-lora \
  --output_json outputs/alpacaeval2_answers.json \
  --judge_model gpt-4o-mini \
  --max_examples 300    # 控制在基准集的一半以下以节省成本
```

MT-Bench（FastChat，建议 mini 评测模型与子集问题数）
```bash
python scripts/eval_mtbench.py --model_dir outputs/llama2-7b-dolly-lora --merged_out outputs/merged-for-fastchat
# Then follow FastChat docs to serve the merged model and run MT-Bench.
# 例如：使用 gpt-4o-mini 并限制 40/80 题
# python -m fastchat.eval.mt_bench --model-path http://localhost:8000 --num-questions 40 --judge-model gpt-4o-mini
```

### 5) Chat with the Model
```bash
python scripts/chat.py --adapter_dir outputs/llama2-7b-dolly-lora --prompt "Explain LoRA in simple terms."
```

### Notes
- The `requirements.txt` pins versions for reproducibility.
- Use `gpt-4o-mini` (or other mini models) for judging to stay within budget.
- Keep evaluation subset sizes (`--max_examples`, `--num_questions`) conservative to fit ~$10 budget.
- PyTorch cu124 wheels are self-contained; a system CUDA 12.8 install is fine.
- For multi-GPU or distributed training, integrate Accelerate/DeepSpeed or FSDP as needed.


