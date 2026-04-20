# classifier_model.py

import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1


class FaceIdentityClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: str = "vggface2",
        freeze_backbone: bool = True,
        dropout_p: float = 0.2,
    ):
        super().__init__()

        # Base embedding network
        self.backbone = InceptionResnetV1(pretrained=pretrained, classify=False)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        embeddings = self.backbone(x)   # [B, 512]
        logits = self.classifier(embeddings)
        return logits
