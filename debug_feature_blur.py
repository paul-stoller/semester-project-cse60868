# debug_feature_blur.py

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from face_parser_bisenet import BiSeNetFaceParser
#from feature_blur import blur_face_except_feature
from feature_blur import preserve_only_feature


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--repo-path", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="resnet18", choices=["resnet18", "resnet34"])
    parser.add_argument("--feature", type=str, default="eyes")
    parser.add_argument("--out", type=str, default="outputs/debug_feature_blur.jpg")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_pil = Image.open(args.image).convert("RGB")

    parser_model = BiSeNetFaceParser(
        repo_path=args.repo_path,
        checkpoint_path=args.checkpoint_path,
        model_name=args.model_name,
        device=device,
    )

    segmentation = parser_model.parse_pil(image_pil)

    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    image_tensor = to_tensor(image_pil)

    '''
    output = blur_face_except_feature(
        image=image_tensor,
        segmentation=segmentation,
        feature_name=args.feature,
        kernel_size=31,
        sigma=10.0,
    )'''

    output = preserve_only_feature(
        image=image_tensor,
        segmentation=segmentation,
        feature_name=args.feature,
        mode="blur",
        gray_value=0.5,
        kernel_size=31,
        sigma=10.0,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    to_pil(output).save(out_path)

    print(f"Saved feature-preserving blur image to: {out_path}")


if __name__ == "__main__":
    main()
