import json
import random
from pathlib import Path
from datasets import load_dataset

PROMPT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Context:\n{context}\n\n"
    "### Response:\n{response}\n"
)

def save_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def main(output_dir="data/dolly15k_prepared", seed=42):
    dataset = load_dataset("databricks/databricks-dolly-15k")
    data = []

    # formatting
    for i in dataset["train"]:
        instruction = (i.get("instruction") or "").strip()
        context = (i.get("context") or "").strip()
        response = (i.get("response") or "").strip()
        if response == "":
            continue 

        formatted_text = PROMPT_TEMPLATE.format(
            instruction=instruction,
            context=context,
            response=response,
        )
        data.append({"text": formatted_text})

    # split dataset
    random.Random(seed).shuffle(data)
    n = len(data)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_split = data[:n_train]
    val_split = data[n_train : n_train + n_val]
    test_split = data[n_train + n_val :]
    splits = {"train": train_split, "validation": val_split, "test": test_split}
 
    out_dir = Path(output_dir)
    save_jsonl(splits["train"], out_dir / "train.jsonl")
    save_jsonl(splits["validation"], out_dir / "validation.jsonl")
    save_jsonl(splits["test"], out_dir / "test.jsonl")
    print(f"Saved splits to {out_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare Dolly-15K with prompt formatting and splits")
    parser.add_argument("--output_dir", type=str, default="data/dolly15k_prepared")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(output_dir=args.output_dir, seed=args.seed)