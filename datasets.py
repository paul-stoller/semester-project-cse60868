# datasets.py

from pathlib import Path
from collections import defaultdict
import random
from typing import Dict, List, Tuple, Set

from PIL import Image
import torch
from torch.utils.data import Dataset


def load_celeba_identity_map(
    img_dir: str,
    identity_file: str
) -> Tuple[Dict[str, int], Dict[int, List[str]]]:
    """
    Reads CelebA identity labels and returns:
      - image_to_identity: {image_name: identity_id}
      - identity_to_images: {identity_id: [full_image_path, ...]}
    """
    img_dir = Path(img_dir)

    image_to_identity: Dict[str, int] = {}
    identity_to_images: Dict[int, List[str]] = defaultdict(list)

    with open(identity_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue

            img_name, identity_str = parts
            identity = int(identity_str)
            full_path = img_dir / img_name

            if not full_path.exists():
                continue

            image_to_identity[img_name] = identity
            identity_to_images[identity].append(str(full_path))

    return image_to_identity, dict(identity_to_images)

def load_lfw_identity_map(
    lfw_root: str
) -> Dict[str, List[str]]:
    """
    Reads LFW folder structure and returns:
      - identity_to_images: {person_name: [full_image_path, ...]}
    """
    lfw_root = Path(lfw_root)
    identity_to_images: Dict[str, List[str]] = {}

    for person_dir in sorted(lfw_root.iterdir()):
        if not person_dir.is_dir():
            continue

        image_paths = sorted(
            [str(p) for p in person_dir.glob("*.jpg")]
        )

        if image_paths:
            identity_to_images[person_dir.name] = image_paths

    return identity_to_images

def split_identities(
    identity_to_images: Dict,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42
) -> Tuple[Set, Set, Set]:
    """
    Splits identities into train/val/test sets.
    No identity appears in more than one split.
    """

    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be less than 1.0")

    identities = sorted(identity_to_images.keys())
    rng = random.Random(seed)
    rng.shuffle(identities)

    n_total = len(identities)
    n_train = int(train_frac * n_total)
    n_val = int(val_frac * n_total)

    train_ids = set(identities[:n_train])
    val_ids = set(identities[n_train:n_train + n_val])
    test_ids = set(identities[n_train + n_val:])

    return train_ids, val_ids, test_ids

class CelebAIdentityDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        identity_file: str,
        split_ids,
        transform=None
    ):
        self.img_dir = Path(img_dir)
        self.transform = transform

        _, identity_to_images = load_celeba_identity_map(
            img_dir=img_dir,
            identity_file=identity_file
        )

        self.samples = []
        for identity in split_ids:
            for img_path in identity_to_images.get(identity, []):
                self.samples.append((img_path, identity))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, identity = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "identity": torch.tensor(identity, dtype=torch.long),
            "img_path": img_path,
        }

class LFWDataset(Dataset):
    def __init__(self, lfw_root: str, transform=None):
        self.transform = transform
        identity_to_images = load_lfw_identity_map(lfw_root)

        self.samples = []
        for identity, image_paths in identity_to_images.items():
            for img_path in image_paths:
                self.samples.append((img_path, identity))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, identity = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "identity": identity,
            "img_path": img_path,
        }
