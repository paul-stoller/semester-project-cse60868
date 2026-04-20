# part4/metrics.py

from typing import Dict, Tuple
import torch
import torch.nn.functional as F


def cosine_distance(emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
    """
    Returns cosine distance for a batch of embeddings.
    """
    sim = F.cosine_similarity(emb1, emb2, dim=1)
    return 1.0 - sim


def compute_accuracy(
    distances: torch.Tensor,
    labels: torch.Tensor,
    threshold: float
) -> float:
    """
    Predict same identity if distance < threshold, else different.
    labels: 1 for same, 0 for different
    """
    preds = (distances < threshold).long()
    correct = (preds == labels).sum().item()
    return correct / len(labels)


def find_best_threshold(
    distances: torch.Tensor,
    labels: torch.Tensor,
    num_thresholds: int = 200
) -> Tuple[float, float]:
    """
    Finds threshold maximizing accuracy on the provided set.
    """
    best_threshold = 0.0
    best_acc = -1.0

    for t in torch.linspace(
        distances.min().item(),
        distances.max().item(),
        steps=num_thresholds
    ):
        acc = compute_accuracy(distances, labels, threshold=t.item())
        if acc > best_acc:
            best_acc = acc
            best_threshold = t.item()

    return best_threshold, best_acc


def summarize_pair_distances(
    distances: torch.Tensor,
    labels: torch.Tensor
) -> Dict[str, float]:
    """
    Returns distance summary stats for positive and negative pairs.
    """
    pos = distances[labels == 1]
    neg = distances[labels == 0]

    return {
        "mean_positive_distance": pos.mean().item() if len(pos) > 0 else float("nan"),
        "mean_negative_distance": neg.mean().item() if len(neg) > 0 else float("nan"),
        "std_positive_distance": pos.std().item() if len(pos) > 1 else 0.0,
        "std_negative_distance": neg.std().item() if len(neg) > 1 else 0.0,
        "num_positive_pairs": int((labels == 1).sum().item()),
        "num_negative_pairs": int((labels == 0).sum().item()),
    }


def compute_confusion_counts(
    distances: torch.Tensor,
    labels: torch.Tensor,
    threshold: float
) -> Dict[str, int]:
    """
    Same/different confusion counts.
    """
    preds = (distances < threshold).long()

    tp = int(((preds == 1) & (labels == 1)).sum().item())
    tn = int(((preds == 0) & (labels == 0)).sum().item())
    fp = int(((preds == 1) & (labels == 0)).sum().item())
    fn = int(((preds == 0) & (labels == 1)).sum().item())

    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
