# inspect_data.py 

from datasets import (
    load_celeba_identity_map,
    load_lfw_identity_map,
    split_identities,
)

def main():
    celeba_img_dir = "data/celeba/img_align_celeba"
    celeba_identity_file = "data/celeba/identity_CelebA.txt"
    lfw_root = "data/lfw/lfw_funneled"

    print("Loading CelebA...")
    image_to_identity, celeba_identity_to_images = load_celeba_identity_map(
        img_dir=celeba_img_dir,
        identity_file=celeba_identity_file
    )

    print(f"CelebA total images indexed: {len(image_to_identity)}")
    print(f"CelebA total identities indexed: {len(celeba_identity_to_images)}")

    train_ids, val_ids, test_ids = split_identities(
        celeba_identity_to_images,
        train_frac=0.6,
        val_frac=0.2,
        seed=42
    )

    print(f"CelebA train identities: {len(train_ids)}")
    print(f"CelebA val identities:   {len(val_ids)}")
    print(f"CelebA test identities:  {len(test_ids)}")

    overlap_train_val = train_ids.intersection(val_ids)
    overlap_train_test = train_ids.intersection(test_ids)
    overlap_val_test = val_ids.intersection(test_ids)

    print(f"Train/Val overlap:  {len(overlap_train_val)}")
    print(f"Train/Test overlap: {len(overlap_train_test)}")
    print(f"Val/Test overlap:   {len(overlap_val_test)}")

    print("\nLoading LFW...")
    lfw_identity_to_images = load_lfw_identity_map(lfw_root)

    print(f"LFW total identities indexed: {len(lfw_identity_to_images)}")
    print(f"LFW total images indexed: {sum(len(v) for v in lfw_identity_to_images.values())}")

    # Show a few sample counts
    print("\nSample CelebA identities:")
    for i, (identity, paths) in enumerate(celeba_identity_to_images.items()):
        print(f"  Identity {identity}: {len(paths)} images")
        if i == 4:
            break

    print("\nSample LFW identities:")
    for i, (identity, paths) in enumerate(lfw_identity_to_images.items()):
        print(f"  Identity {identity}: {len(paths)} images")
        if i == 4:
            break


if __name__ == "__main__":
    main()
