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
        # Training loss entries have both "loss" and "learning_rate"
        if "loss" in entry and "learning_rate" in entry:
            step = entry.get("step", len(steps))
            steps.append(step)
            train_losses.append(entry["loss"])
        # Evaluation loss entries have "eval_loss"
        if "eval_loss" in entry:
            step = entry.get("step", 0)
            # If step is not present, use epoch or try to infer from last training step
            if step == 0:
                if "epoch" in entry:
                    # Approximate: assume steps are evenly distributed
                    step = int(entry.get("epoch", 0) * len(steps) / max(len(steps), 1))
                else:
                    step = len(eval_steps) * (steps[-1] // max(len(steps), 1)) if steps else len(eval_steps)
            eval_steps.append(step)
            eval_losses.append(entry["eval_loss"])

    Path(Path(out_png).parent).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    if steps and train_losses:
        plt.plot(steps, train_losses, label="Train Loss", alpha=0.7, linewidth=1.5)
        print(f"📊 Training loss: {len(train_losses)} data points")
    
    if eval_steps and eval_losses:
        plt.plot(eval_steps, eval_losses, label="Validation Loss", marker="o", markersize=4, linewidth=1.5, alpha=0.8)
        print(f"📊 Validation loss: {len(eval_losses)} data points")
    
    if not steps and not eval_steps:
        print("⚠️  No loss data found in trainer_state.json")
        return
    
    plt.xlabel("Training Steps", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.title("Training and Validation Loss Curves", fontsize=13, fontweight="bold")
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"✅ Saved loss plot: {out_png}")
    
    # Print summary statistics
    if train_losses:
        print(f"   Train loss: {min(train_losses):.4f} → {train_losses[-1]:.4f}")
    if eval_losses:
        print(f"   Valid loss: {min(eval_losses):.4f} → {eval_losses[-1]:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot training and validation loss from Trainer logs")
    parser.add_argument("--trainer_state", type=str, default="outputs/llama2-7b-dolly-lora/trainer_state.json")
    parser.add_argument("--out_png", type=str, default="outputs/loss_curve.png")
    args = parser.parse_args()
    main(args.trainer_state, args.out_png)


