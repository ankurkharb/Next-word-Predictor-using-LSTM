"""
Plot model training curves after running train_model.py.

Usage:
    python plot_training.py
"""

import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


HISTORY_PATH = os.path.join("models", "training_history.json")
OUTPUT_PATH = os.path.join("eda_output", "08_training_curves.png")


def main():
    if not os.path.exists(HISTORY_PATH):
        raise FileNotFoundError("Missing models/training_history.json. Run `python train_model.py` first.")

    os.makedirs("eda_output", exist_ok=True)

    with open(HISTORY_PATH, "r", encoding="utf-8") as history_file:
        history = json.load(history_file)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["loss"], label="Train Loss", color="royalblue", linewidth=2)
    ax1.plot(history["val_loss"], label="Val Loss", color="tomato", linewidth=2)
    ax1.set_title("Model Loss", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(history["accuracy"], label="Train Acc", color="green", linewidth=2)
    ax2.plot(history["val_accuracy"], label="Val Acc", color="orange", linewidth=2)
    ax2.set_title("Model Accuracy", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.suptitle("LSTM Training History", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
