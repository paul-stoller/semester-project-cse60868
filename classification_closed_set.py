# classification_closed_set.py

from collections import defaultdict
from typing import Dict, List, Tuple
import random

from PIL import Image
import torch
from torch.utils.data import Dataset

from datasets import load_celeba_identity_map


def select_top_k_identities(
    identity_to_images: Dict[int, List[str]],
    top_k: int = 50,
    min_images: int = 20
) -> Dict[int, List[str]]:
    """
    Keep only identities with at least `min_images`, then select top_k by image count.
    """
    filtered = {
        identity: images
        for identity, images in identity_to_images.items()
        if len(images) >= min_images
    }

    sorted_items = sorted(
        filtered.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )[:top_k]

    return dict(sorted_items)


def build_closed_set_classification_splits(
    img_dir: str,
    identity_file: str,
    top_k: int = 50,
    min_images: int = 20,
    train_frac: float = 0.7,
    seed: int = 42
):
    """
    For closed-set classification:
    - Use the same identities in train and val
    - Split images within each identity
    """
    _, identity_to_images = load_celeba_identity_map(
        img_dir=img_dir,
        identity_file=identity_file
    )

    selected = select_top_k_identities(
        identity_to_images=identity_to_images,
        top_k=top_k,
        min_images=min_images
    )

    label_map = {identity: idx for idx, identity in enumerate(sorted(selected.keys()))}

    rng = random.Random(seed)

    train_samples = []
    val_samples = []

    for identity, images in selected.items():
        images = list(images)
        rng.shuffle(images)

        n_train = int(len(images) * train_frac)
        train_imgs = images[:n_train]
        val_imgs = images[n_train:]

        class_label = label_map[identity]

        for img_path in train_imgs:
            train_samples.append((img_path, class_label))

        for img_path in val_imgs:
            val_samples.append((img_path, class_label))

    return train_samples, val_samples, label_map


class FaceClassificationDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "img_path": img_path,
        }
