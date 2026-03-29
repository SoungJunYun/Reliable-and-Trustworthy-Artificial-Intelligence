import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class SimpleMNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28 -> 14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14 -> 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class NormalizedModel(nn.Module):
    """
    입력 x는 항상 [0, 1] 범위 raw image라고 가정.
    모델 내부에서만 normalize 수행.
    """
    def __init__(self, base_model, mean, std):
        super().__init__()
        self.base_model = base_model
        self.register_buffer("mean", torch.tensor(mean).view(1, len(mean), 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, len(std), 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.base_model(x)


def build_cifar10_model(use_pretrained=True, freeze_backbone=False):
    weights = ResNet18_Weights.DEFAULT if use_pretrained else None
    base_model = resnet18(weights=weights)

    if freeze_backbone:
        for param in base_model.parameters():
            param.requires_grad = False

    in_features = base_model.fc.in_features
    base_model.fc = nn.Linear(in_features, 10)

    # freeze_backbone=True일 때 마지막 fc만 학습되게 설정
    if freeze_backbone:
        for param in base_model.fc.parameters():
            param.requires_grad = True

    # CIFAR-10 raw image [0,1] -> model 내부 normalize
    return NormalizedModel(
        base_model,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )