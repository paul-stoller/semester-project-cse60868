import random
import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import (
    load_celeba_identity_map,
    split_identities,
    CelebAIdentityDataset,
)
from transforms_utils import get_eval_transform
from models import load_frozen_face_model, get_embeddings


def apply_random_occlusion(images: torch.Tensor, patch_frac: float = 0.2, fill_value: float = 1.0):
    """
    Applies a random square occlusion to each image in a batch.
    images: [B, C, H, W]
    """
    occluded = images.clone()
    batch_size, channels, height, width = occluded.shape

    patch_h = max(1, int(height * patch_frac))
    patch_w = max(1, int(width * patch_frac))

    for i in range(batch_size):
        top = random.randint(0, height - patch_h)
        left = random.randint(0, width - patch_w)
        occluded[i, :, top:top + patch_h, left:left + patch_w] = fill_value

    return occluded


def cosine_distance(clean_emb: torch.Tensor, pert_emb: torch.Tensor) -> torch.Tensor:
    sim = F.cosine_similarity(clean_emb, pert_emb, dim=1)
    return 1.0 - sim


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    celeba_img_dir = "data/celeba/img_align_celeba"
    identity_file = "data/celeba/identity_CelebA.txt"

    _, identity_to_images = load_celeba_identity_map(
        img_dir=celeba_img_dir,
        identity_file=identity_file
    )

    train_ids, val_ids, test_ids = split_identities(
        identity_to_images,
        train_frac=0.6,
        val_frac=0.2,
        seed=42
    )

    eval_transform = get_eval_transform(image_size=160)

    val_dataset = CelebAIdentityDataset(
        img_dir=celeba_img_dir,
        identity_file=identity_file,
        split_ids=val_ids,
        transform=eval_transform
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    model = load_frozen_face_model(device=device)

    mean_distances = []


    max_batches = 100 # roughly 1600 images
    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
        if batch_idx >= max_batches:
            break
        images = batch["image"].to(device)

        occluded_images = apply_random_occlusion(
            images,
            patch_frac=0.2,
            fill_value=1.0
        )

        clean_emb = get_embeddings(model, images)
        occluded_emb = get_embeddings(model, occluded_images)

        distances = cosine_distance(clean_emb, occluded_emb)
        mean_distances.append(distances.mean().item())

        if batch_idx == 0:
            print(f"First batch image tensor shape: {images.shape}")
            print(f"First batch mean cosine distance: {distances.mean().item():.6f}")

    overall_mean = sum(mean_distances) / len(mean_distances)
    print(f"Validation batches evaluated: {len(mean_distances)}")
    print(f"Overall mean embedding shift under random occlusion: {overall_mean:.6f}")


if __name__ == "__main__":
    main()
