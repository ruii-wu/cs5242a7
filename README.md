## LLaMA-2-7B LoRA on Dolly-15K (PEFT)

### Setup
Python 3.10+
- Install in two steps to avoid indexing issues:
```bash
python3 -m venv venv

source venv/bin/activate

pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124

#Install the remaining dependencies from PyPI
pip install -r requirements.txt

# hugging login
huggingface-cli login
```

### 1) Prepare Dataset
```bash
python codes/prep_dataset.py --output_dir data/dolly15k_prepared
```

### 2) Train (LoRA)
```bash
python codes/train.py \
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

AlpacaEval 2 outputs:
```bash
# Automatically loads prompts from alpaca-eval package
python scripts/eval_alpacaeval2.py \
  --base_model meta-llama/Llama-2-7b-hf \
  --adapter_dir outputs/llama2-7b-dolly-lora/checkpoint-2250 \
  --output_dir outputs/eval_alpacaeval2 \
  --run_base \
  --max_examples 300 \
  --max_new_tokens 512 --temperature 0.2 --top_p 0.95

# Judge and compare
alpaca_eval evaluate \
  --model_outputs outputs/eval_alpacaeval2/alpacaeval2_finetuned_outputs.json outputs/eval_alpacaeval2/alpacaeval2_base_outputs.json \
  --annotators_config gpt-4o-mini
```

MT-Bench answers:
```bash
python scripts/eval_mtbench.py \
  --base_model meta-llama/Llama-2-7b-hf \
  --adapter_dir outputs/llama2-7b-dolly-lora/checkpoint-2250 \
  --output_dir data/mt_bench/model_answer \
  --run_base \
  --max_questions 40 \
  --max_new_tokens 512 --temperature 0.2 --top_p 0.95

# Judge with FastChat and summarize
python -m fastchat.llm_judge.gen_judgment \
    --judge-model gpt-4o-mini \
    --mode pairwise-all \
    --model-list llama-2-7b-hf llama2-7b-dolly-lora \
    --parallel 4 \
    --first-n 40 \
    --baseline-model gpt-4

# Show results
python -m fastchat.llm_judge.show_result \
    --judge-model gpt-4o-mini \
    --mode pairwise-all \
    --model-list llama-2-7b-hf llama2-7b-dolly-lora
```