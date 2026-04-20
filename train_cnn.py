# train_cnn.py

from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from classification_closed_set import (
    build_closed_set_classification_splits,
    FaceClassificationDataset,
)
from cnn_model import SmallFaceCNN
from transforms_utils import get_train_transform, get_eval_transform


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)

        total_loss += loss.item() * images.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            preds = torch.argmax(logits, dim=1)

            total_loss += loss.item() * images.size(0)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    celeba_img_dir = "data/celeba/img_align_celeba"
    identity_file = "data/celeba/identity_CelebA.txt"

    # Hyperparameters (IMPORTANT)
    image_size = 160
    batch_size = 32
    num_epochs = 25
    learning_rate = 1e-4
    top_k = 20
    min_images = 20

    train_samples, val_samples, label_map = build_closed_set_classification_splits(
        img_dir=celeba_img_dir,
        identity_file=identity_file,
        top_k=top_k,
        min_images=min_images,
        train_frac=0.7,
    )

    num_classes = len(label_map)

    train_dataset = FaceClassificationDataset(
        samples=train_samples,
        transform=get_train_transform(image_size=image_size)
    )

    val_dataset = FaceClassificationDataset(
        samples=val_samples,
        transform=get_eval_transform(image_size=image_size)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SmallFaceCNN(num_classes=num_classes).to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        print(f"Train Acc: {train_acc:.4f}")
        print(f"Val Acc:   {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_map": label_map,
                    "image_size": image_size,
                    "best_val_acc": best_val_acc,
                },
                output_dir / "cnn_from_scratch.pt"
            )

    print(f"\nBest Val Acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
