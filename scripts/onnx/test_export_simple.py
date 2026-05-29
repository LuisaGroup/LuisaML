#!/usr/bin/env python3
"""Simple test script that exports a tiny MLP from PyTorch to ONNX."""

import torch
import torch.nn as nn
import os

class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def main():
    model = TinyMLP()
    model.eval()
    
    dummy_input = torch.randn(1, 4)
    output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "onnx", "data")
    os.makedirs(output_dir, exist_ok=True)
    
    onnx_path = os.path.join(output_dir, "simple_mlp.onnx")
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=21,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    
    print(f"Exported ONNX model to: {onnx_path}")
    
    # Try to simplify with onnxsim if available
    try:
        import onnxsim
        import onnx
        model_onnx = onnx.load(onnx_path)
        model_simplified, check = onnxsim.simplify(model_onnx)
        if check:
            simplified_path = os.path.join(output_dir, "simple_mlp_simplified.onnx")
            onnx.save(model_simplified, simplified_path)
            print(f"Simplified ONNX model saved to: {simplified_path}")
        else:
            print("ONNX simplification failed validation.")
    except ImportError:
        print("onnxsim not available, skipping simplification.")
    
    print("Export test completed successfully.")

if __name__ == "__main__":
    main()
