import json
from pathlib import Path

import matplotlib.pyplot as plt


def main(trainer_state_path: str = "outputs/llama2-7b-dolly-lora/trainer_state.json", out_png: str = "outputs/loss_curve.png"):
    with open(trainer_state_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    logs = state.get("log_history", [])
    steps = []
    train_losses = []
    eval_steps = []
    eval_losses = []

    for entry in logs:
        if "loss" in entry and "learning_rate" in entry:
            steps.append(entry.get("step", len(steps)))
            train_losses.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(entry.get("step", len(eval_steps)))
            eval_losses.append(entry["eval_loss"])

    Path(Path(out_png).parent).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    if steps:
        plt.plot(steps, train_losses, label="train loss")
    if eval_steps:
        plt.plot(eval_steps, eval_losses, label="eval loss")
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.title("Training/Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print("Saved:", out_png)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot training and validation loss from Trainer logs")
    parser.add_argument("--trainer_state", type=str, default="outputs/llama2-7b-dolly-lora/trainer_state.json")
    parser.add_argument("--out_png", type=str, default="outputs/loss_curve.png")
    args = parser.parse_args()
    main(args.trainer_state, args.out_png)


