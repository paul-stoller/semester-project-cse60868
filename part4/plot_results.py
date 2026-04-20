# plot_results.py

import json
from pathlib import Path
import matplotlib.pyplot as plt


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_resolution_accuracy():
    verification_files = [
        ("112", "outputs/verification_112.json"),
        ("160", "outputs/verification_160.json"),
        ("224", "outputs/verification_224.json"),
    ]

    robustness_files = [
        ("112", "outputs/robustness_112.json"),
        ("160", "outputs/robustness_160.json"),
        ("224", "outputs/robustness_224.json"),
    ]

    clean_x = []
    train_acc = []
    val_acc = []
    pert_val_acc = []

    for res, path in verification_files:
        if Path(path).exists():
            d = load_json(path)
            clean_x.append(int(res))
            train_acc.append(d["train_accuracy"])
            val_acc.append(d["validation_accuracy"])

    for res, path in robustness_files:
        if Path(path).exists():
            d = load_json(path)
            pert_val_acc.append((int(res), d["validation_perturbed_accuracy"]))

    if clean_x:
        plt.figure(figsize=(8, 5))
        plt.plot(clean_x, train_acc, marker="o", label="Train verification accuracy")
        plt.plot(clean_x, val_acc, marker="o", label="Validation verification accuracy")

        if pert_val_acc:
            px = [x for x, _ in pert_val_acc]
            py = [y for _, y in pert_val_acc]
            plt.plot(px, py, marker="o", label="Validation perturbed accuracy")

        plt.xlabel("Image Resolution")
        plt.ylabel("Accuracy")
        plt.title("Verification Accuracy vs Resolution")
        plt.legend()
        Path("outputs").mkdir(exist_ok=True)
        plt.savefig("outputs/resolution_accuracy_plot.png", bbox_inches="tight")
        plt.close()
        print("Saved plot to outputs/resolution_accuracy_plot.png")
    else:
        print("No JSON result files found for plotting.")


if __name__ == "__main__":
    plot_resolution_accuracy()
