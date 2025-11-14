"""CIFAR-10 Dataset and Vision Encoder Utilities"""

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import CIFAR10


class CIFAR10VisionEncoder:
    """
    Helper class for loading CIFAR-10 and encoding images to features.
    Reusable across different optimization experiments.
    """

    def __init__(self, dim: int = 512, data_root: str = "./data", device: str = "cuda"):
        self.dim = dim
        self.data_root = data_root
        self.device = device

        # Transform for CIFAR-10 images
        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Load CIFAR-10 dataset
        self.dataset = CIFAR10(root=self.data_root, train=False, download=True, transform=self.transform)

        # Simple vision encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((24, 24)),
            nn.Flatten(1),
            nn.Linear(64 * 24 * 24, dim),
        ).to(device)

        print(f"CIFAR-10 loaded: {len(self.dataset)} images")

    def get_batch(self, batch_size: int, start_idx: int = 0) -> torch.Tensor:
        """Get a batch of images as tensors"""
        batch_images = []
        for i in range(batch_size):
            idx = (start_idx + i) % len(self.dataset)
            img, _ = self.dataset[idx]
            batch_images.append(img)
        return torch.stack(batch_images).to(self.device)

    def encode_batch(self, batch_size: int, start_idx: int = 0, num_tokens: int = 576) -> torch.Tensor:
        """
        Get a batch of images and encode them to visual features.

        Args:
            batch_size: Number of images in batch
            start_idx: Starting index in dataset
            num_tokens: Number of visual tokens to expand to

        Returns:
            [batch_size, num_tokens, dim]
        """
        batch_tensor = self.get_batch(batch_size, start_idx)

        with torch.no_grad():
            visual_features = self.encoder(batch_tensor)
            visual_features = visual_features.view(batch_size, 1, self.dim)
            visual_features = visual_features.expand(-1, num_tokens, -1).contiguous()

        return visual_features
