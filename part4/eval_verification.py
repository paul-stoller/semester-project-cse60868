# part4/eval_verification.py

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from tqdm import tqdm

from datasets import load_celeba_identity_map, split_identities
from models import load_frozen_face_model, get_embeddings
from transforms_utils import get_eval_transform
from part4.pairs import sample_balanced_pairs
from part4.metrics import (
    cosine_distance,
    compute_accuracy,
    find_best_threshold,
    summarize_pair_distances,
    compute_confusion_counts,
)


def load_single_image(path: str, transform):
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0)


@torch.no_grad()
def compute_pair_distances(
    model,
    pairs: List[Tuple[str, str, int]],
    transform,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    distances = []
    labels = []

    for img1_path, img2_path, label in tqdm(pairs, desc="Computing pair distances"):
        x1 = load_single_image(img1_path, transform).to(device)
        x2 = load_single_image(img2_path, transform).to(device)

        emb1 = get_embeddings(model, x1)
        emb2 = get_embeddings(model, x2)

        dist = cosine_distance(emb1, emb2)
        distances.append(dist.item())
        labels.append(label)

    return torch.tensor(distances), torch.tensor(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--num-positive", type=int, default=1000)
    parser.add_argument("--num-negative", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", type=str, default="outputs/verification_results.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    celeba_img_dir = "data/celeba/img_align_celeba"
    identity_file = "data/celeba/identity_CelebA.txt"

    _, identity_to_images = load_celeba_identity_map(
        img_dir=celeba_img_dir,
        identity_file=identity_file
    )

    train_ids, val_ids, _ = split_identities(
        identity_to_images,
        train_frac=0.6,
        val_frac=0.2,
        seed=args.seed
    )

    train_map: Dict = {k: identity_to_images[k] for k in train_ids}
    val_map: Dict = {k: identity_to_images[k] for k in val_ids}

    transform = get_eval_transform(image_size=args.image_size)
    model = load_frozen_face_model(device=device)

    print("Sampling training pairs...")
    train_pairs = sample_balanced_pairs(
        identity_to_images=train_map,
        num_positive=args.num_positive,
        num_negative=args.num_negative,
        seed=args.seed
    )

    print("Sampling validation pairs...")
    val_pairs = sample_balanced_pairs(
        identity_to_images=val_map,
        num_positive=args.num_positive,
        num_negative=args.num_negative,
        seed=args.seed + 10
    )

    print("\nEvaluating training pairs...")
    train_distances, train_labels = compute_pair_distances(
        model=model,
        pairs=train_pairs,
        transform=transform,
        device=device
    )

    threshold, train_acc = find_best_threshold(
        distances=train_distances,
        labels=train_labels
    )

    train_summary = summarize_pair_distances(train_distances, train_labels)
    train_conf = compute_confusion_counts(train_distances, train_labels, threshold)

    print("\nEvaluating validation pairs...")
    val_distances, val_labels = compute_pair_distances(
        model=model,
        pairs=val_pairs,
        transform=transform,
        device=device
    )

    val_acc = compute_accuracy(
        distances=val_distances,
        labels=val_labels,
        threshold=threshold
    )
    val_summary = summarize_pair_distances(val_distances, val_labels)
    val_conf = compute_confusion_counts(val_distances, val_labels, threshold)

    results = {
        "image_size": args.image_size,
        "threshold": threshold,
        "train_accuracy": train_acc,
        "validation_accuracy": val_acc,
        "train_summary": train_summary,
        "validation_summary": val_summary,
        "train_confusion": train_conf,
        "validation_confusion": val_conf,
    }

    print("\n===== Verification Results =====")
    print(f"Image size:            {args.image_size}")
    print(f"Best train threshold:  {threshold:.6f}")
    print(f"Train accuracy:        {train_acc:.4f}")
    print(f"Validation accuracy:   {val_acc:.4f}")
    print(f"Train mean pos dist:   {train_summary['mean_positive_distance']:.4f}")
    print(f"Train mean neg dist:   {train_summary['mean_negative_distance']:.4f}")
    print(f"Val mean pos dist:     {val_summary['mean_positive_distance']:.4f}")
    print(f"Val mean neg dist:     {val_summary['mean_negative_distance']:.4f}")

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()
