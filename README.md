## LLaMA-2-7B LoRA on Dolly-15K (PEFT)

### Setup
- Python 3.10+ recommended
- Install with CUDA 12.x-compatible PyTorch wheels (cu124 self-contained):

```bash
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu124
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
  --epochs 3 --batch_size 1 --grad_accum_steps 16 \
  --max_length 1024 --lr 2e-4
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

AlpacaEval 2 (requires `OPENAI_API_KEY` for judged metrics):
```bash
set OPENAI_API_KEY=sk-...   # Windows PowerShell: $env:OPENAI_API_KEY="sk-..."
python scripts/eval_alpacaeval2.py --model_dir outputs/llama2-7b-dolly-lora --output_json outputs/alpacaeval2_answers.json
```

MT-Bench (FastChat):
```bash
python scripts/eval_mtbench.py --model_dir outputs/llama2-7b-dolly-lora --merged_out outputs/merged-for-fastchat
# Then follow FastChat docs to serve the merged model and run MT-Bench.
```

### 5) Chat with the Model
```bash
python scripts/chat.py --adapter_dir outputs/llama2-7b-dolly-lora --prompt "Explain LoRA in simple terms."
```

### Notes
- The `requirements.txt` pins versions for reproducibility.
- PyTorch cu124 wheels are self-contained; a system CUDA 12.8 install is fine.
- For multi-GPU or distributed training, integrate Accelerate/DeepSpeed or FSDP as needed.


