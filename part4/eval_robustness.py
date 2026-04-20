# eval_robustness.py

import argparse
import json
import random
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
)


def load_single_image(path: str, transform):
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0)


def apply_random_occlusion(
    images: torch.Tensor,
    patch_frac: float = 0.2,
    fill_value: float = 1.0
) -> torch.Tensor:
    occluded = images.clone()
    batch_size, channels, height, width = occluded.shape

    patch_h = max(1, int(height * patch_frac))
    patch_w = max(1, int(width * patch_frac))

    for i in range(batch_size):
        top = random.randint(0, height - patch_h)
        left = random.randint(0, width - patch_w)
        occluded[i, :, top:top + patch_h, left:left + patch_w] = fill_value

    return occluded


@torch.no_grad()
def compute_pair_distances(
    model,
    pairs: List[Tuple[str, str, int]],
    transform,
    device: str,
    apply_perturbation: bool = False,
    patch_frac: float = 0.2
):
    distances = []
    labels = []

    for img1_path, img2_path, label in tqdm(pairs, desc="Computing robustness distances"):
        x1 = load_single_image(img1_path, transform).to(device)
        x2 = load_single_image(img2_path, transform).to(device)

        if apply_perturbation:
            x1 = apply_random_occlusion(x1, patch_frac=patch_frac)
            x2 = apply_random_occlusion(x2, patch_frac=patch_frac)

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
    parser.add_argument("--patch-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", type=str, default="outputs/robustness_results.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(args.seed)

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

    train_pairs = sample_balanced_pairs(
        identity_to_images=train_map,
        num_positive=args.num_positive,
        num_negative=args.num_negative,
        seed=args.seed
    )

    val_pairs = sample_balanced_pairs(
        identity_to_images=val_map,
        num_positive=args.num_positive,
        num_negative=args.num_negative,
        seed=args.seed + 10
    )

    print("Computing clean training threshold...")
    train_clean_dist, train_clean_labels = compute_pair_distances(
        model=model,
        pairs=train_pairs,
        transform=transform,
        device=device,
        apply_perturbation=False,
        patch_frac=args.patch_frac
    )

    threshold, train_clean_acc = find_best_threshold(
        distances=train_clean_dist,
        labels=train_clean_labels
    )

    print("\nEvaluating clean validation pairs...")
    val_clean_dist, val_clean_labels = compute_pair_distances(
        model=model,
        pairs=val_pairs,
        transform=transform,
        device=device,
        apply_perturbation=False,
        patch_frac=args.patch_frac
    )
    val_clean_acc = compute_accuracy(val_clean_dist, val_clean_labels, threshold)

    print("\nEvaluating perturbed validation pairs...")
    val_pert_dist, val_pert_labels = compute_pair_distances(
        model=model,
        pairs=val_pairs,
        transform=transform,
        device=device,
        apply_perturbation=True,
        patch_frac=args.patch_frac
    )
    val_pert_acc = compute_accuracy(val_pert_dist, val_pert_labels, threshold)

    clean_summary = summarize_pair_distances(val_clean_dist, val_clean_labels)
    pert_summary = summarize_pair_distances(val_pert_dist, val_pert_labels)

    results = {
        "image_size": args.image_size,
        "patch_frac": args.patch_frac,
        "threshold": threshold,
        "train_clean_accuracy": train_clean_acc,
        "validation_clean_accuracy": val_clean_acc,
        "validation_perturbed_accuracy": val_pert_acc,
        "validation_clean_summary": clean_summary,
        "validation_perturbed_summary": pert_summary,
    }

    print("\n===== Robustness Results =====")
    print(f"Image size:                    {args.image_size}")
    print(f"Patch fraction:                {args.patch_frac}")
    print(f"Threshold (from clean train):  {threshold:.6f}")
    print(f"Train clean accuracy:          {train_clean_acc:.4f}")
    print(f"Validation clean accuracy:     {val_clean_acc:.4f}")
    print(f"Validation perturbed accuracy: {val_pert_acc:.4f}")
    print(f"Clean mean pos dist:           {clean_summary['mean_positive_distance']:.4f}")
    print(f"Clean mean neg dist:           {clean_summary['mean_negative_distance']:.4f}")
    print(f"Pert mean pos dist:            {pert_summary['mean_positive_distance']:.4f}")
    print(f"Pert mean neg dist:            {pert_summary['mean_negative_distance']:.4f}")

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()
