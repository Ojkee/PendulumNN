import torch

from torch import nn


class NeuralNetwork(nn.Module):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {DEVICE} device")

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, output_dim),
        )
