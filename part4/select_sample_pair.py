# select_sample_pair.py

import shutil
from pathlib import Path

from datasets import load_celeba_identity_map, split_identities


def main():
    celeba_img_dir = "data/celeba/img_align_celeba"
    identity_file = "data/celeba/identity_CelebA.txt"

    _, identity_to_images = load_celeba_identity_map(
        img_dir=celeba_img_dir,
        identity_file=identity_file
    )

    _, val_ids, _ = split_identities(
        identity_to_images,
        train_frac=0.6,
        val_frac=0.2,
        seed=42
    )

    # Find first validation identity with at least 2 images
    selected_identity = None
    selected_images = None
    for identity in sorted(val_ids):
        images = identity_to_images[identity]
        if len(images) >= 2:
            selected_identity = identity
            selected_images = images[:2]
            break

    if selected_images is None:
        raise ValueError("No validation identity with at least 2 images was found.")

    out_dir = Path("samples")
    out_dir.mkdir(parents=True, exist_ok=True)

    out1 = out_dir / "sample_pair_1.jpg"
    out2 = out_dir / "sample_pair_2.jpg"

    shutil.copy(selected_images[0], out1)
    shutil.copy(selected_images[1], out2)

    print(f"Selected validation identity: {selected_identity}")
    print(f"Copied sample pair to:\n  {out1}\n  {out2}")


if __name__ == "__main__":
    main()
