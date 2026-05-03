# face_parser_bisenet.py

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


class BiSeNetFaceParser:
    def __init__(
        self,
        repo_path: str,
        checkpoint_path: str,
        model_name: str = "resnet18",
        device: str = "cpu",
        input_size: int = 512,
    ):
        self.repo_path = Path(repo_path).expanduser().resolve()
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()

        if not self.repo_path.exists():
            raise FileNotFoundError(f"face-parsing repo not found: {self.repo_path}")

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        # IMPORTANT:
        # Put face-parsing repo at the FRONT of sys.path so that
        # "from models.bisenet import BiSeNet" resolves to:
        # ~/Desktop/face-parsing/models/bisenet.py
        # instead of this project's models.py.
        repo_path_str = str(self.repo_path)
        if repo_path_str in sys.path:
            sys.path.remove(repo_path_str)
        sys.path.insert(0, repo_path_str)

        # If Python already imported this project's models.py,
        # remove it so face-parsing can import its own models package.
        if "models" in sys.modules:
            loaded_models = sys.modules["models"]
            if not hasattr(loaded_models, "__path__"):
                del sys.modules["models"]

        from inference import load_model

        self.device = torch.device(device)
        self.input_size = input_size

        self.model = load_model(
            model_name=model_name,
            num_classes=19,
            weight_path=str(self.checkpoint_path),
            device=self.device,
        )
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @torch.no_grad()
    def parse_pil(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        original_w, original_h = image.size

        x = self.transform(image).unsqueeze(0).to(self.device)

        output = self.model(x)

        if isinstance(output, (tuple, list)):
            output = output[0]

        predicted = torch.argmax(output, dim=1).float().unsqueeze(1)

        predicted = F.interpolate(
            predicted,
            size=(original_h, original_w),
            mode="nearest",
        )

        return predicted.squeeze(0).squeeze(0).long().cpu()
