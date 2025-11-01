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

# When running mtbench, another version of openai library needs to be installed, running following command:
# pip install openai==0.28

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
python codes/plot_losses.py \
  --trainer_state outputs/llama2-7b-dolly-lora/checkpoint-2250/trainer_state.json \
  --out_png outputs/loss_curve_final.png
```

### 4) Evaluation (AlpacaEval 2 + MT-Bench)
Export the OPENAI_API_KEY first

AlpacaEval 2 outputs:
```bash
# Automatically loads prompts from alpaca-eval package
python codes/eval_alpacaeval2.py \
  --base_model meta-llama/Llama-2-7b-hf \
  --adapter_dir outputs/llama2-7b-dolly-lora/checkpoint-2250 \
  --output_dir outputs/eval_alpacaeval2 \
  --run_base \
  --max_examples 300 \
  --max_new_tokens 512 --temperature 0.2 --top_p 0.95
```

# Judge and compare

If gpt-4o-mini is used, its configs.yaml needs to be added to ./venv/lib/python3.12/site-packages/alpaca_eval/evaluators_configs/gpt-4o-mini/configs.yaml

gpt-4o-mini:
  prompt_template: "chatgpt/basic_prompt.txt"
  fn_completions: "openai_completions"
  completions_kwargs:
    model_name: "gpt-4o-mini"
    max_tokens: 50
    temperature: 0
  completion_parser_kwargs:
    outputs_to_match:
      1: '(?:^|\n) ?Output \(a\)'
      2: '(?:^|\n) ?Output \(b\)'
  batch_size: 1
'''

```bash
alpaca_eval evaluate \
  --model_outputs outputs/eval_alpacaeval2/alpacaeval2_finetuned_outputs.json outputs/eval_alpacaeval2/alpacaeval2_base_outputs.json \
  --annotators_config gpt-4o-mini
```

MT-Bench answers:
```bash
python codes/eval_mtbench.py \
  --base_model meta-llama/Llama-2-7b-hf \
  --adapter_dir outputs/llama2-7b-dolly-lora/checkpoint-2250 \
  --output_dir data/mt_bench/model_answer \
  --run_base \
  --max_questions 40 \
  --max_new_tokens 512 --temperature 0.2 --top_p 0.95
```

# Judge with FastChat and summarize
If gpt-4o-mini is used, its model name needs to be added to OPENAI_MODEL_LIST in the ./venv/lib/python3.12/site-packages/fastchat/model/model_adapter.py

OPENAI_MODEL_LIST = (
    "gpt-4o-mini"
)

And openai library version needs to be openai==0.28

```bash
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