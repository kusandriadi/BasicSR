"""
Model statistics utility for lightweight model validation
"""

import time
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_model_size_mb(model: nn.Module) -> float:
    """Calculate model size in MB (assuming float32)"""
    param_count = count_parameters(model)
    return param_count * 4 / (1024**2)  # 4 bytes per float32


def measure_inference_speed(model: nn.Module, input_size: Tuple[int, int, int, int] = (1, 3, 128, 128), 
                          device: str = 'cuda', num_runs: int = 100) -> Tuple[float, float]:
    """
    Measure inference speed
    
    Returns:
        Tuple[avg_time_ms, fps]
    """
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
    """
    Measure GPU memory usage during forward pass
    
    Returns:
        memory_used_mb (float)
    """
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
    """
    Calculate architecture complexity metrics
    
    Returns:
        Dict with complexity metrics
    """
    conv_count = 0
    bn_count = 0
    linear_count = 0
    attention_count = 0
    activation_count = 0
    
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            conv_count += 1
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
            bn_count += 1
        elif isinstance(module, nn.Linear):
            linear_count += 1
        elif isinstance(module, (nn.MultiheadAttention, nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
            attention_count += 1
        elif isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.GELU, nn.ELU, nn.Sigmoid, nn.Tanh)):
            activation_count += 1
    
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
        'activation_layers': activation_count,
        'total_layers': conv_count + bn_count + linear_count + attention_count
    }


def print_model_lightweight_stats(model: nn.Module, model_name: str = "Model", 
                                input_size: Tuple[int, int, int, int] = (1, 3, 128, 128),
                                device: str = 'cuda') -> Dict:
    """
    Print comprehensive lightweight model statistics
    
    Args:
        model: PyTorch model
        model_name: Name for display
        input_size: Input tensor size (B, C, H, W)
        device: Device for speed/memory testing
        
    Returns:
        Dict with all statistics
    """
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
    
    # Return all stats for logging
    return {
        'model_name': model_name,
        'parameters': param_count,
        'model_size_mb': model_size_mb,
        'inference_time_ms': avg_time_ms,
        'fps': fps,
        'gpu_memory_mb': gpu_memory_mb,
        'architecture_complexity': complexity
    }


def detect_input_size_from_config(opt: Optional[dict] = None) -> Tuple[int, int, int, int]:
    """
    Detect appropriate input size from training config
    
    Args:
        opt: Training options dictionary
        
    Returns:
        Tuple[batch, channels, height, width]
    """
    if opt is None:
        return (1, 3, 128, 128)  # Default fallback
    
    batch_size = 1  # Always use batch=1 for stats
    
    # Try to get channels from config
    channels = 3  # Default RGB
    if 'datasets' in opt and 'train' in opt['datasets']:
        train_dataset = opt['datasets']['train']
        if 'num_in_ch' in train_dataset:
            channels = train_dataset['num_in_ch']
        elif 'io_backend' in train_dataset and 'grayscale' in str(train_dataset.get('io_backend', {})):
            channels = 1
    
    # Try to get spatial size from config
    height = width = 128  # Default size
    
    # Priority order for size detection:
    # 1. GT size (most accurate for discriminator)
    # 2. Crop size
    # 3. Network input size
    # 4. Scale * LQ size
    
    if 'datasets' in opt and 'train' in opt['datasets']:
        train_dataset = opt['datasets']['train']
        
        # Method 1: GT size (best for discriminator)
        if 'gt_size' in train_dataset:
            height = width = train_dataset['gt_size']
        
        # Method 2: Crop size 
        elif 'crop_size' in train_dataset:
            if isinstance(train_dataset['crop_size'], (list, tuple)):
                height, width = train_dataset['crop_size'][:2]
            else:
                height = width = train_dataset['crop_size']
        
        # Method 3: Calculate from LQ size and scale
        elif all(k in train_dataset for k in ['lq_size', 'scale']):
            lq_size = train_dataset['lq_size']
            scale = train_dataset['scale']
            if isinstance(lq_size, (list, tuple)):
                height, width = [s * scale for s in lq_size[:2]]
            else:
                height = width = lq_size * scale
    
    # Try network config as fallback
    if height == 128 and width == 128:  # Still default
        if 'network_g' in opt and 'scale' in opt['network_g']:
            scale = opt['network_g']['scale']
            # Assume common LQ sizes
            common_lq_sizes = [64, 32, 48]  # Common LQ sizes
            for lq_size in common_lq_sizes:
                if lq_size * scale in [128, 256, 512]:  # Common GT sizes
                    height = width = lq_size * scale
                    break
    
    return (batch_size, channels, height, width)


def log_discriminator_stats(discriminator: nn.Module, 
                          discriminator_name: str = "Discriminator",
                          input_size: Optional[Tuple[int, int, int, int]] = None,
                          opt: Optional[dict] = None) -> None:
    """
    Specialized function for discriminator statistics logging
    Designed to be called at the end of training
    
    Args:
        discriminator: Discriminator model
        discriminator_name: Name for display
        input_size: Optional input size tuple (B,C,H,W). If None, will auto-detect
        opt: Optional training config for auto-detection
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Auto-detect input size if not provided
    if input_size is None:
        input_size = detect_input_size_from_config(opt)
        print(f"Auto-detected input size: {input_size}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED - DISCRIMINATOR EFFICIENCY REPORT")
    print("="*80)
    
    stats = print_model_lightweight_stats(
        model=discriminator,
        model_name=discriminator_name,
        input_size=input_size,
        device=device
    )
    
    # Additional efficiency indicators
    params = stats['parameters']
    fps = stats['fps']
    gpu_mem = stats['gpu_memory_mb']
    
    print("\nEFFICIENCY INDICATORS:")
    
    # Parameter efficiency
    if params < 100_000:
        print("✅ Ultra-lightweight: < 100K parameters")
    elif params < 500_000:
        print("✅ Lightweight: < 500K parameters")
    elif params < 1_000_000:
        print("⚠️  Moderate: < 1M parameters")
    else:
        print("❌ Heavy: > 1M parameters")
    
    # Speed efficiency
    if fps > 2000:
        print("🚀 Ultra-fast: > 2000 FPS")
    elif fps > 1000:
        print("⚡ Fast: > 1000 FPS")
    elif fps > 500:
        print("🔥 Moderate: > 500 FPS")
    else:
        print("🐌 Slow: < 500 FPS")
    
    # Memory efficiency
    if gpu_mem < 1.0:
        print("💾 Ultra-efficient: < 1 MB GPU memory")
    elif gpu_mem < 5.0:
        print("💾 Efficient: < 5 MB GPU memory")
    elif gpu_mem < 20.0:
        print("💾 Moderate: < 20 MB GPU memory")
    else:
        print("💾 Heavy: > 20 MB GPU memory")
    
    print("="*80)
    
    return stats