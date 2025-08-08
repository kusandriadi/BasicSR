#!/usr/bin/env python3
"""
Standalone test for model stats (without full basicsr import)
"""

import torch
import torch.nn as nn
import time
from typing import Dict, Tuple


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_model_size_mb(model: nn.Module) -> float:
    """Calculate model size in MB (assuming float32)"""
    param_count = count_parameters(model)
    return param_count * 4 / (1024**2)


def measure_inference_speed(model: nn.Module, input_size: Tuple[int, int, int, int] = (1, 3, 128, 128), 
                          device: str = 'cuda', num_runs: int = 100) -> Tuple[float, float]:
    """Measure inference speed"""
    if not torch.cuda.is_available() and device == 'cuda':
        device = 'cpu'
        
    model = model.to(device)
    model.eval()
    input_tensor = torch.randn(input_size, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)
    
    # Measure
    if device == 'cuda':
        torch.cuda.synchronize()
        
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)
    
    if device == 'cuda':
        torch.cuda.synchronize()
        
    end_time = time.time()
    
    avg_time_ms = (end_time - start_time) * 1000 / num_runs
    fps = 1000 / avg_time_ms
    
    return avg_time_ms, fps


def measure_gpu_memory_mb(model: nn.Module, input_size: Tuple[int, int, int, int] = (1, 3, 128, 128)) -> float:
    """Measure GPU memory usage during forward pass"""
    if not torch.cuda.is_available():
        return 0.0
        
    device = 'cuda'
    model = model.to(device)
    input_tensor = torch.randn(input_size, device=device)
    
    # Clear cache and reset stats
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Measure memory before
    mem_before = torch.cuda.memory_allocated(device) / 1024**2
    
    # Forward pass
    with torch.no_grad():
        _ = model(input_tensor)
    
    # Memory after
    mem_after = torch.cuda.memory_allocated(device) / 1024**2
    
    return mem_after - mem_before


def calculate_architecture_complexity(model: nn.Module) -> Dict[str, int]:
    """Calculate architecture complexity metrics"""
    conv_count = 0
    bn_count = 0
    linear_count = 0
    attention_count = 0
    
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            conv_count += 1
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
            bn_count += 1
        elif isinstance(module, nn.Linear):
            linear_count += 1
    
    # Check for attention-like patterns in custom modules
    for name, module in model.named_modules():
        module_name = type(module).__name__.lower()
        if any(keyword in module_name for keyword in ['attention', 'transformer', 'multihead']):
            attention_count += 1
    
    return {
        'conv_layers': conv_count,
        'batch_norm_layers': bn_count,
        'linear_layers': linear_count,
        'attention_layers': attention_count,
        'total_layers': conv_count + bn_count + linear_count + attention_count
    }


def print_model_lightweight_stats(model: nn.Module, model_name: str = "Model", 
                                input_size: Tuple[int, int, int, int] = (1, 3, 128, 128),
                                device: str = 'cuda') -> Dict:
    """Print comprehensive lightweight model statistics"""
    print("=" * 80)
    print(f"LIGHTWEIGHT MODEL STATISTICS: {model_name}")
    print("=" * 80)
    
    # 1. Parameter Count
    param_count = count_parameters(model)
    print(f"Parameters: {param_count:,}")
    
    # 2. Model Size
    model_size_mb = calculate_model_size_mb(model)
    print(f"Model Size: {model_size_mb:.2f} MB")
    
    # 3. Speed Measurement
    try:
        avg_time_ms, fps = measure_inference_speed(model, input_size, device)
        print(f"Inference Speed: {avg_time_ms:.2f} ms ({fps:.1f} FPS)")
    except Exception as e:
        print(f"Speed measurement failed: {e}")
        avg_time_ms, fps = 0.0, 0.0
    
    # 4. GPU Memory Usage
    try:
        gpu_memory_mb = measure_gpu_memory_mb(model, input_size)
        print(f"GPU Memory Usage: {gpu_memory_mb:.2f} MB")
    except Exception as e:
        print(f"GPU memory measurement failed: {e}")
        gpu_memory_mb = 0.0
    
    # 5. Architecture Complexity
    complexity = calculate_architecture_complexity(model)
    print("Architecture Complexity:")
    print(f"  - Conv Layers: {complexity['conv_layers']}")
    print(f"  - BatchNorm Layers: {complexity['batch_norm_layers']}")
    print(f"  - Linear Layers: {complexity['linear_layers']}")
    print(f"  - Attention Layers: {complexity['attention_layers']}")
    print(f"  - Total Layers: {complexity['total_layers']}")
    
    print("=" * 80)
    
    return {
        'model_name': model_name,
        'parameters': param_count,
        'model_size_mb': model_size_mb,
        'inference_time_ms': avg_time_ms,
        'fps': fps,
        'gpu_memory_mb': gpu_memory_mb,
        'architecture_complexity': complexity
    }


# Simple test discriminator
class SimpleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.conv3 = nn.Conv2d(128, 1, 4, 1, 1)
        self.act = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return self.conv3(x)


if __name__ == "__main__":
    print("Testing Model Stats Functions (Standalone)")
    print("=" * 60)
    
    # Test with simple discriminator
    model = SimpleDiscriminator()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    stats = print_model_lightweight_stats(
        model=model,
        model_name="Simple Test Discriminator",
        input_size=(1, 3, 128, 128),
        device=device
    )
    
    print(f"\nTest completed successfully!")
    print(f"Device used: {device}")
    print(f"Final stats: {stats['parameters']:,} params, {stats['fps']:.1f} FPS")