# eval_feature_blur_classifier.py

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from classification_closed_set import (
    build_closed_set_classification_splits,
    FaceClassificationDataset,
)
from cnn_model import SmallFaceCNN
from face_parser_bisenet import BiSeNetFaceParser
from feature_blur import preserve_only_feature
from transforms_utils import get_eval_transform


def load_cnn_checkpoint(checkpoint_path, num_classes, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = SmallFaceCNN(num_classes=num_classes).to(device)

    # Handles both checkpoint styles:
    # 1. {"model_state_dict": ...}
    # 2. raw model.state_dict()
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def normalize_tensor(image_tensor):
    """
    Matches transforms_utils normalization:
    ToTensor gives [0,1], then Normalize([0.5]*3, [0.5]*3).
    """
    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    )
    return normalize(image_tensor)


@torch.no_grad()
def evaluate_clean(model, samples, device, image_size, max_samples=None):
    correct = 0
    total = 0

    eval_transform = get_eval_transform(image_size=image_size)

    if max_samples is not None:
        samples = samples[:max_samples]

    for img_path, label in tqdm(samples, desc="Clean evaluation"):
        image = Image.open(img_path).convert("RGB")
        x = eval_transform(image).unsqueeze(0).to(device)
        y = torch.tensor([label], dtype=torch.long).to(device)

        logits = model(x)
        pred = torch.argmax(logits, dim=1)

        correct += (pred == y).sum().item()
        total += 1

    return correct / total


@torch.no_grad()
def evaluate_feature_only(
    model,
    parser_model,
    samples,
    feature_name,
    device,
    image_size,
    max_samples=None,
):
    correct = 0
    total = 0

    resize = transforms.Resize((image_size, image_size))
    to_tensor = transforms.ToTensor()

    if max_samples is not None:
        samples = samples[:max_samples]

    for img_path, label in tqdm(samples, desc=f"Feature-only: {feature_name}"):
        image_pil = Image.open(img_path).convert("RGB")

        # Resize first so parser mask and classifier input match.
        image_pil = resize(image_pil)

        # BiSeNet segmentation on resized image.
        segmentation = parser_model.parse_pil(image_pil)

        # Tensor in [0,1], before classifier normalization.
        image_tensor = to_tensor(image_pil)

        # Preserve only selected feature; blur everything else.
        feature_only = preserve_only_feature(
            image=image_tensor,
            segmentation=segmentation,
            feature_name=feature_name,
            mode="blur",
            gray_value=0.5,
            kernel_size=31,
            sigma=10.0,
        )

        # Normalize for classifier.
        x = normalize_tensor(feature_only).unsqueeze(0).to(device)
        y = torch.tensor([label], dtype=torch.long).to(device)

        logits = model(x)
        pred = torch.argmax(logits, dim=1)

        correct += (pred == y).sum().item()
        total += 1

    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-path", type=str, required=True)
    parser.add_argument("--parser-checkpoint", type=str, required=True)
    parser.add_argument("--classifier-checkpoint", type=str, default="outputs/cnn_from_scratch.pt")
    parser.add_argument("--model-name", type=str, default="resnet18", choices=["resnet18", "resnet34"])

    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--min-images", type=int, default=20)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--max-samples", type=int, default=200)

    parser.add_argument(
        "--features",
        nargs="+",
        default=["eyes", "eyebrows", "nose", "mouth", "hair", "skin"],
    )

    parser.add_argument("--out-json", type=str, default="outputs/feature_blur_results.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    celeba_img_dir = "data/celeba/img_align_celeba"
    identity_file = "data/celeba/identity_CelebA.txt"

    train_samples, val_samples, label_map = build_closed_set_classification_splits(
        img_dir=celeba_img_dir,
        identity_file=identity_file,
        top_k=args.top_k,
        min_images=args.min_images,
        train_frac=args.train_frac,
    )

    num_classes = len(label_map)

    print(f"Number of classes: {num_classes}")
    print(f"Validation samples available: {len(val_samples)}")
    print(f"Max samples evaluated: {args.max_samples}")

    model = load_cnn_checkpoint(
        checkpoint_path=args.classifier_checkpoint,
        num_classes=num_classes,
        device=device,
    )

    parser_model = BiSeNetFaceParser(
        repo_path=args.repo_path,
        checkpoint_path=args.parser_checkpoint,
        model_name=args.model_name,
        device=device,
    )

    results = {}

    clean_acc = evaluate_clean(
        model=model,
        samples=val_samples,
        device=device,
        image_size=args.image_size,
        max_samples=args.max_samples,
    )

    results["clean_accuracy"] = clean_acc

    print("\n===== Feature-Only Blur Results =====")
    print(f"Clean accuracy: {clean_acc:.4f}")

    for feature in args.features:
        acc = evaluate_feature_only(
            model=model,
            parser_model=parser_model,
            samples=val_samples,
            feature_name=feature,
            device=device,
            image_size=args.image_size,
            max_samples=args.max_samples,
        )

        results[f"{feature}_only_accuracy"] = acc
        results[f"{feature}_accuracy_drop"] = clean_acc - acc

        print(f"{feature:10s} accuracy: {acc:.4f} | drop: {clean_acc - acc:.4f}")

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()
