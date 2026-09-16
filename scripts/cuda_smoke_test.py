"""CUDA Smoke Test for GeoFUSE SentinelGuard.

Verifies:
1. PyTorch version, CUDA availability, CUDA version, GPU name, and total VRAM.
2. Moves ResidualSRNet model and 4-band input tensor batch to cuda:0.
3. Executes forward pass, loss computation, backward pass, and optimizer step on GPU.
4. Asserts all model parameters, inputs, outputs, and losses reside on cuda:0.
5. Measures and prints GPU memory allocation.
"""

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from src.models.model import ResidualSRNet, build_model
from src.models.loss import CompoundSRLoss
from src.utils.config import load_config


def run_cuda_smoke_test() -> bool:
    print("=" * 70)
    print("       GeoFUSE SentinelGuard — CUDA Hardware & Pipeline Smoke Test")
    print("=" * 70)

    # 1. Environment & CUDA Information
    torch_version = torch.__version__
    cuda_available = torch.cuda.is_available()
    cuda_version = torch.version.cuda
    device_count = torch.cuda.device_count() if cuda_available else 0

    print(f"PyTorch Version       : {torch_version}")
    print(f"CUDA Available        : {cuda_available}")
    print(f"CUDA Runtime Version  : {cuda_version}")
    print(f"Available Devices     : {device_count}")

    if not cuda_available:
        print("\n[FAIL] CUDA is NOT available to PyTorch. Aborting smoke test.")
        return False

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)

    print(f"Active Device         : {device}")
    print(f"GPU Name              : {gpu_name}")
    print(f"Total GPU VRAM        : {total_memory:.1f} MiB")
    print("-" * 70)

    # 2. Model Instantiation & Device Movement
    print("[1/5] Building ResidualSRNet model...")
    config = load_config()
    model = build_model(config).to(device)
    print(f"      Model parameter count: {sum(p.numel() for p in model.parameters()):,}")

    for name, param in model.named_parameters():
        assert param.device.type == "cuda", f"Parameter {name} is on {param.device}, expected 'cuda'"
    print("      Assert Passed: All model parameters are located on cuda:0.")

    # 3. Input & Target Tensor Creation on CUDA
    print("[2/5] Creating synthetic 4-band input & target tensors on cuda:0...")
    batch_size = 4
    channels = 4
    lr_size = 64
    hr_size = 128

    lr_tensor = torch.rand((batch_size, channels, lr_size, lr_size), dtype=torch.float32, device=device)
    hr_target = torch.rand((batch_size, channels, hr_size, hr_size), dtype=torch.float32, device=device)

    assert lr_tensor.device.type == "cuda", f"Input tensor is on {lr_tensor.device}, expected 'cuda'"
    assert hr_target.device.type == "cuda", f"Target tensor is on {hr_target.device}, expected 'cuda'"
    print(f"      Input Tensor Shape   : {tuple(lr_tensor.shape)} on {lr_tensor.device}")
    print(f"      Target Tensor Shape  : {tuple(hr_target.shape)} on {hr_target.device}")

    # 4. Forward Pass on CUDA
    print("[3/5] Executing forward pass on cuda:0...")
    sr_output = model(lr_tensor)
    assert sr_output.device.type == "cuda", f"Output tensor is on {sr_output.device}, expected 'cuda'"
    assert sr_output.shape == (batch_size, channels, hr_size, hr_size), f"Unexpected output shape: {sr_output.shape}"
    print(f"      Forward pass successful: Output shape {tuple(sr_output.shape)} on {sr_output.device}")

    # 5. Loss Computation & Backward Pass on CUDA
    print("[4/5] Executing loss computation & backward pass on cuda:0...")
    criterion = CompoundSRLoss(channels=channels, grad_weight=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    optimizer.zero_grad()
    loss, loss_dict = criterion(sr_output, hr_target)
    assert loss.device.type == "cuda", f"Loss tensor is on {loss.device}, expected 'cuda'"
    print(f"      Computed Compound Loss: {loss.item():.5f} (L1: {loss_dict['l1']:.5f}, Grad: {loss_dict['grad']:.5f})")

    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient after backward pass"
            assert param.grad.device.type == "cuda", f"Gradient for {name} is on {param.grad.device}, expected 'cuda'"
    print("      Backward pass successful: All parameter gradients reside on cuda:0.")

    optimizer.step()
    print("      Optimizer step successful.")

    # 6. GPU Memory Telemetry
    print("[5/5] Querying CUDA memory statistics...")
    allocated_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)
    max_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    print(f"      Allocated Memory     : {allocated_mb:.2f} MiB")
    print(f"      Reserved Memory      : {reserved_mb:.2f} MiB")
    print(f"      Peak Allocated Memory: {max_allocated_mb:.2f} MiB")
    print("-" * 70)
    print("   [SUCCESS] CUDA SMOKE TEST PASSED: All hardware, model, tensor, and gradient checks verified on cuda:0!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = run_cuda_smoke_test()
    sys.exit(0 if success else 1)
