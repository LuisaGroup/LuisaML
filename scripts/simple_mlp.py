"""
Simple MLP neural network for ONNX export testing.
"""
import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """A simple feed-forward MLP with configurable hidden layers."""

    def __init__(self, input_dim: int = 784, hidden_dim: int = 256,
                 output_dim: int = 10, num_hidden: int = 2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_hidden):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def create_model(input_dim: int = 784, hidden_dim: int = 256,
                 output_dim: int = 10, num_hidden: int = 2) -> SimpleMLP:
    """Factory to create a SimpleMLP instance."""
    return SimpleMLP(input_dim, hidden_dim, output_dim, num_hidden)


def get_sample_input(input_dim: int = 784, batch_size: int = 1) -> torch.Tensor:
    """Return a sample input tensor for tracing/export."""
    return torch.randn(batch_size, input_dim)


if __name__ == "__main__":
    model = create_model()
    x = get_sample_input()
    y = model(x)
    print(f"Model output shape: {y.shape}")
    print(f"Model params: {sum(p.numel() for p in model.parameters())}")
