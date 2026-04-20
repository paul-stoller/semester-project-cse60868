# train_classifier_closed_set.py

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
from classifier_model import FaceIdentityClassifier
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

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


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

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths
    celeba_img_dir = "data/celeba/img_align_celeba"
    identity_file = "data/celeba/identity_CelebA.txt"

    # Hyperparameters
    image_size = 160
    batch_size = 32
    num_epochs = 5
    learning_rate = 1e-3
    weight_decay = 1e-4
    top_k = 50
    min_images = 20
    freeze_backbone = True
    seed = 42

    # Data
    train_samples, val_samples, label_map = build_closed_set_classification_splits(
        img_dir=celeba_img_dir,
        identity_file=identity_file,
        top_k=top_k,
        min_images=min_images,
        train_frac=0.7,
        seed=seed
    )

    num_classes = len(label_map)

    print(f"Number of classes: {num_classes}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples:   {len(val_samples)}")

    train_dataset = FaceClassificationDataset(
        samples=train_samples,
        transform=get_train_transform(image_size=image_size)
    )

    val_dataset = FaceClassificationDataset(
        samples=val_samples,
        transform=get_eval_transform(image_size=image_size)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # Model
    model = FaceIdentityClassifier(
        num_classes=num_classes,
        pretrained="vggface2",
        freeze_backbone=freeze_backbone,
        dropout_p=0.2
    ).to(device)

    # Only optimize trainable parameters
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    criterion = nn.CrossEntropyLoss()

    # Output dir
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0
    best_model_path = output_dir / "best_classifier.pt"

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        val_loss, val_acc = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_map": label_map,
                    "num_classes": num_classes,
                    "image_size": image_size,
                    "top_k": top_k,
                    "min_images": min_images,
                    "best_val_acc": best_val_acc,
                },
                best_model_path
            )
            print(f"Saved new best model to {best_model_path}")

    history_path = output_dir / "classifier_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("\nTraining complete.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Saved history to {history_path}")


if __name__ == "__main__":
    main()
