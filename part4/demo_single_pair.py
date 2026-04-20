# demo_single_pair.py

import argparse
from PIL import Image
import torch

from models import load_frozen_face_model, get_embeddings
from transforms_utils import get_eval_transform
from part4.metrics import cosine_distance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img1", type=str, default="samples/sample_pair_1.jpg")
    parser.add_argument("--img2", type=str, default="samples/sample_pair_2.jpg")
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--threshold", type=float, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_frozen_face_model(device=device)
    transform = get_eval_transform(image_size=args.image_size)

    x1 = transform(Image.open(args.img1).convert("RGB")).unsqueeze(0).to(device)
    x2 = transform(Image.open(args.img2).convert("RGB")).unsqueeze(0).to(device)

    emb1 = get_embeddings(model, x1)
    emb2 = get_embeddings(model, x2)

    dist = cosine_distance(emb1, emb2).item()
    prediction = "same identity" if dist < args.threshold else "different identity"

    print("===== Single Pair Demo =====")
    print(f"Image 1:      {args.img1}")
    print(f"Image 2:      {args.img2}")
    print(f"Image size:   {args.image_size}")
    print(f"Distance:     {dist:.6f}")
    print(f"Threshold:    {args.threshold:.6f}")
    print(f"Prediction:   {prediction}")


if __name__ == "__main__":
    main()
