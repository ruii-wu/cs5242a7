import json
import random
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset


PROMPT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Context:\n{context}\n\n"
    "### Response:\n{response}\n"
)


def format_example(example: Dict[str, str]) -> Dict[str, str]:
    instruction = (example.get("instruction") or "").strip()
    context = (example.get("context") or "").strip()
    response = (example.get("response") or "").strip()

    # Drop rows with empty response
    if response == "":
        return {}

    formatted_text = PROMPT_TEMPLATE.format(
        instruction=instruction,
        context=context,
        response=response,
    )
    return {"text": formatted_text}


def split_dataset(items: List[Dict[str, str]], seed: int = 42) -> Dict[str, List[Dict[str, str]]]:
    random.Random(seed).shuffle(items)
    n = len(items)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train = items[:n_train]
    val = items[n_train : n_train + n_val]
    test = items[n_train + n_val :]
    return {"train": train, "validation": val, "test": test}


def save_jsonl(rows: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(output_dir: str = "data/dolly15k_prepared", seed: int = 42) -> None:
    print("Loading dataset databricks/databricks-dolly-15k ...")
    ds = load_dataset("databricks/databricks-dolly-15k")
    records: List[Dict[str, str]] = []
    for ex in ds["train"]:
        formatted = format_example(ex)
        if formatted:
            records.append(formatted)

    print(f"Total usable examples: {len(records)}")
    splits = split_dataset(records, seed=seed)

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


