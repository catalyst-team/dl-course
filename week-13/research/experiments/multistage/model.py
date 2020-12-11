import torch
from torch import nn


class SimpleNet(nn.Module):
    def __init__(
        self,
        num_filters1: int = 6,
        num_filters2: int = 16,
        num_hiddens1: int = 120,
        num_hiddens2: int = 84,
        num_classes: int = 10,
    ):
        super().__init__()
        self.num_hiddens0 = num_filters2 * 5 * 5

        self.core = nn.Sequential(
            nn.Conv2d(3, num_filters1, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(num_filters1, num_filters2, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(self.num_hiddens0, num_hiddens1),
            nn.ReLU(),
            nn.Linear(num_hiddens1, num_hiddens2),
            nn.ReLU(),
        )

        self.head = nn.Sequential(nn.Linear(num_hiddens2, num_classes),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.core(x)
        return self.head(output)
