# train_baseline.py

from datasets import (
    load_celeba_identity_map,
    split_identities,
    CelebAIdentityDataset,
)
from transforms_utils import get_train_transform, get_eval_transform


def main():
    celeba_img_dir = "data/celeba/img_align_celeba"
    identity_file = "data/celeba/identity_CelebA.txt"

    train_transform = get_train_transform(image_size=160)
    val_transform = get_eval_transform(image_size=160)

    _, identity_to_images = load_celeba_identity_map(
        img_dir=celeba_img_dir,
        identity_file=identity_file
    )

    # Split identities (not images) to prevent identity leakage across subsets
    train_ids, val_ids, test_ids = split_identities(
        identity_to_images,
        train_frac=0.6,
        val_frac=0.2,
        seed=42
    )

    print(f"Train identities: {len(train_ids)}")
    print(f"Val identities:   {len(val_ids)}")
    print(f"Test identities:  {len(test_ids)}")

    assert len(train_ids & val_ids) == 0
    assert len(train_ids & test_ids) == 0
    assert len(val_ids & test_ids) == 0

    train_dataset = CelebAIdentityDataset(
        img_dir=celeba_img_dir,
        identity_file=identity_file,
        split_ids=train_ids,
        transform=train_transform
    )

    val_dataset = CelebAIdentityDataset(
        img_dir=celeba_img_dir,
        identity_file=identity_file,
        split_ids=val_ids,
        transform=val_transform
    )

    test_dataset = CelebAIdentityDataset(
        img_dir=celeba_img_dir,
        identity_file=identity_file,
        split_ids=test_ids,
        transform=val_transform
    )

    print("\nDataset sizes:")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size:   {len(val_dataset)}")
    print(f"Test dataset size:  {len(test_dataset)}")

    sample = train_dataset[0]
    print(f"\nSample image shape: {sample['image'].shape}")
    print(f"Sample identity: {sample['identity']}")


if __name__ == "__main__":
    main()
