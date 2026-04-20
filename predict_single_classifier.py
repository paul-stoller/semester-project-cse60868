# predict_single_classifier.py

import argparse
import torch
from PIL import Image

from cnn_model import SmallFaceCNN
from transforms_utils import get_eval_transform


def load_model(checkpoint_path, num_classes, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = SmallFaceCNN(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint["label_map"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="outputs/cnn_from_scratch.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    label_map = checkpoint["label_map"]
    num_classes = len(label_map)

    model = SmallFaceCNN(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    transform = get_eval_transform(image_size=checkpoint["image_size"])

    image = Image.open(args.image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image)
        pred_class = torch.argmax(logits, dim=1).item()

    # Reverse label map
    inv_label_map = {v: k for k, v in label_map.items()}

    print(f"Predicted identity: {inv_label_map[pred_class]}")


if __name__ == "__main__":
    main()
