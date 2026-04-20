# pairs.py

import random
from typing import Dict, List, Tuple


Pair = Tuple[str, str, int]  # (img1, img2, label), label=1 same, 0 different


def filter_identities_with_min_images(
    identity_to_images: Dict,
    min_images: int = 2
) -> Dict:
    """
    Keeps only identities with at least `min_images` available.
    Needed for positive pair generation.
    """
    return {
        identity: images
        for identity, images in identity_to_images.items()
        if len(images) >= min_images
    }


def sample_positive_pairs(
    identity_to_images: Dict,
    num_pairs: int = 1000,
    seed: int = 42
) -> List[Pair]:
    """
    Samples positive pairs (same identity).
    """
    rng = random.Random(seed)
    valid_map = filter_identities_with_min_images(identity_to_images, min_images=2)
    identities = list(valid_map.keys())

    if not identities:
        raise ValueError("No identities with at least 2 images were found.")

    pairs: List[Pair] = []

    while len(pairs) < num_pairs:
        identity = rng.choice(identities)
        images = valid_map[identity]
        img1, img2 = rng.sample(images, 2)
        pairs.append((img1, img2, 1))

    return pairs


def sample_negative_pairs(
    identity_to_images: Dict,
    num_pairs: int = 1000,
    seed: int = 42
) -> List[Pair]:
    """
    Samples negative pairs (different identities).
    """
    rng = random.Random(seed)
    identities = list(identity_to_images.keys())

    if len(identities) < 2:
        raise ValueError("Need at least 2 identities to sample negative pairs.")

    pairs: List[Pair] = []

    while len(pairs) < num_pairs:
        id1, id2 = rng.sample(identities, 2)
        img1 = rng.choice(identity_to_images[id1])
        img2 = rng.choice(identity_to_images[id2])
        pairs.append((img1, img2, 0))

    return pairs


def sample_balanced_pairs(
    identity_to_images: Dict,
    num_positive: int = 1000,
    num_negative: int = 1000,
    seed: int = 42
) -> List[Pair]:
    """
    Returns a balanced mixed list of positive and negative pairs.
    """
    pos_pairs = sample_positive_pairs(
        identity_to_images=identity_to_images,
        num_pairs=num_positive,
        seed=seed
    )
    neg_pairs = sample_negative_pairs(
        identity_to_images=identity_to_images,
        num_pairs=num_negative,
        seed=seed + 1
    )

    all_pairs = pos_pairs + neg_pairs
    rng = random.Random(seed + 2)
    rng.shuffle(all_pairs)
    return all_pairs
