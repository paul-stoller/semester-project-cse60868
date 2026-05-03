# feature_blur.py

from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F


FACE_PARSING_LABELS: Dict[str, int] = {
    "background": 0,
    "skin": 1,
    "left_eyebrow": 2,
    "right_eyebrow": 3,
    "left_eye": 4,
    "right_eye": 5,
    "eyeglass": 6,
    "left_ear": 7,
    "right_ear": 8,
    "earring": 9,
    "nose": 10,
    "mouth": 11,
    "upper_lip": 12,
    "lower_lip": 13,
    "neck": 14,
    "necklace": 15,
    "cloth": 16,
    "hair": 17,
    "hat": 18,
}


FEATURE_GROUPS: Dict[str, List[str]] = {
    "eyes": ["left_eye", "right_eye"],
    "eyebrows": ["left_eyebrow", "right_eyebrow"],
    "nose": ["nose"],
    "mouth": ["mouth", "upper_lip", "lower_lip"],
    "ears": ["left_ear", "right_ear"],
    "hair": ["hair"],
    "skin": ["skin"],
}


FACE_REGION_LABELS: List[str] = [
    "skin",
    "left_eyebrow",
    "right_eyebrow",
    "left_eye",
    "right_eye",
    "eyeglass",
    "left_ear",
    "right_ear",
    "nose",
    "mouth",
    "upper_lip",
    "lower_lip",
    "hair",
]


def labels_to_mask(segmentation: torch.Tensor, label_names: Iterable[str]) -> torch.Tensor:
    """
    segmentation: [H, W] tensor of integer parsing labels.
    Returns: [1, H, W] boolean mask.
    """
    mask = torch.zeros_like(segmentation, dtype=torch.bool)

    for name in label_names:
        label_id = FACE_PARSING_LABELS[name]
        mask |= segmentation == label_id

    return mask.unsqueeze(0)


def gaussian_blur_tensor(
    image: torch.Tensor,
    kernel_size: int = 31,
    sigma: float = 10.0,
) -> torch.Tensor:
    """
    image: [C, H, W], values assumed in [0, 1].
    Returns blurred image with same shape.
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")

    device = image.device
    channels, height, width = image.shape

    coords = torch.arange(kernel_size, device=device) - kernel_size // 2
    gauss_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss_1d = gauss_1d / gauss_1d.sum()

    kernel_2d = gauss_1d[:, None] @ gauss_1d[None, :]
    kernel_2d = kernel_2d.expand(channels, 1, kernel_size, kernel_size)

    image_batch = image.unsqueeze(0)
    blurred = F.conv2d(
        image_batch,
        kernel_2d,
        padding=kernel_size // 2,
        groups=channels,
    )

    return blurred.squeeze(0)


def blur_face_except_feature(
    image: torch.Tensor,
    segmentation: torch.Tensor,
    feature_name: str,
    kernel_size: int = 31,
    sigma: float = 10.0,
) -> torch.Tensor:
    """
    Blurs the face/head region except for the selected facial feature.

    image: [C, H, W], values in [0, 1]
    segmentation: [H, W], integer labels from face parser
    feature_name: one of FEATURE_GROUPS keys, e.g. "eyes", "nose", "mouth"
    """
    if feature_name not in FEATURE_GROUPS:
        raise ValueError(f"Unknown feature '{feature_name}'. Options: {list(FEATURE_GROUPS)}")

    blurred = gaussian_blur_tensor(
        image=image,
        kernel_size=kernel_size,
        sigma=sigma,
    )

    feature_mask = labels_to_mask(segmentation, FEATURE_GROUPS[feature_name]).to(image.device)
    face_mask = labels_to_mask(segmentation, FACE_REGION_LABELS).to(image.device)

    # Start with original image.
    output = image.clone()

    # Blur the face/head region.
    output = torch.where(face_mask, blurred, output)

    # Restore selected feature.
    output = torch.where(feature_mask, image, output)

    return output.clamp(0.0, 1.0)

def preserve_only_feature(
    image: torch.Tensor,
    segmentation: torch.Tensor,
    feature_name: str,
    mode: str = "gray",
    gray_value: float = 0.5,
    kernel_size: int = 31,
    sigma: float = 10.0,
) -> torch.Tensor:
    """
    Keeps only the selected feature visible.
    Everything else is replaced by either:
      - gray fill
      - blurred image
      - black fill

    image: [C, H, W], values in [0, 1]
    segmentation: [H, W], integer parsing labels
    feature_name: e.g. "eyes", "nose", "mouth"
    mode: "gray", "blur", or "black"
    """
    if feature_name not in FEATURE_GROUPS:
        raise ValueError(f"Unknown feature '{feature_name}'. Options: {list(FEATURE_GROUPS)}")

    feature_mask = labels_to_mask(segmentation, FEATURE_GROUPS[feature_name]).to(image.device)

    if mode == "gray":
        base = torch.full_like(image, fill_value=gray_value)
    elif mode == "black":
        base = torch.zeros_like(image)
    elif mode == "blur":
        base = gaussian_blur_tensor(image=image, kernel_size=kernel_size, sigma=sigma)
    else:
        raise ValueError("mode must be one of: 'gray', 'black', 'blur'")

    output = torch.where(feature_mask, image, base)
    return output.clamp(0.0, 1.0)
