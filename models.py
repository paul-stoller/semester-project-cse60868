import torch
from facenet_pytorch import InceptionResnetV1


def load_frozen_face_model(device: str = "cpu"):
    """
    Loads a pretrained face embedding model and freezes all parameters.
    """
    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    for param in model.parameters():
        param.requires_grad = False

    return model


@torch.no_grad()
def get_embeddings(model, images: torch.Tensor) -> torch.Tensor:
    """
    Returns embeddings for a batch of images.
    """
    return model(images)
