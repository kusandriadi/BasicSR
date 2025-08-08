#!/usr/bin/env python3
"""
Test dynamic input size detection dari config
"""

import sys
sys.path.append('.')

from typing import Optional, Tuple

def detect_input_size_from_config(opt: Optional[dict] = None) -> Tuple[int, int, int, int]:
    """
    Detect appropriate input size from training config
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
            common_lq_sizes = [64, 32, 48]
            for lq_size in common_lq_sizes:
                if lq_size * scale in [128, 256, 512]:
                    height = width = lq_size * scale
                    break
    
    return (batch_size, channels, height, width)


def test_input_size_detection():
    """Test different config scenarios"""
    
    test_cases = [
        # Case 1: No config (default)
        {
            "name": "No Config",
            "config": None,
            "expected": (1, 3, 128, 128)
        },
        
        # Case 2: Standard RealESRGAN config
        {
            "name": "RealESRGAN 256x256",
            "config": {
                "datasets": {
                    "train": {
                        "gt_size": 256,
                        "num_in_ch": 3
                    }
                }
            },
            "expected": (1, 3, 256, 256)
        },
        
        # Case 3: High resolution config
        {
            "name": "High-Res 512x512",
            "config": {
                "datasets": {
                    "train": {
                        "gt_size": 512,
                        "num_in_ch": 3
                    }
                }
            },
            "expected": (1, 3, 512, 512)
        },
        
        # Case 4: Grayscale config
        {
            "name": "Grayscale 128x128",
            "config": {
                "datasets": {
                    "train": {
                        "gt_size": 128,
                        "num_in_ch": 1
                    }
                }
            },
            "expected": (1, 1, 128, 128)
        },
        
        # Case 5: Crop size config
        {
            "name": "Crop Size 384x384",
            "config": {
                "datasets": {
                    "train": {
                        "crop_size": 384,
                        "num_in_ch": 3
                    }
                }
            },
            "expected": (1, 3, 384, 384)
        },
        
        # Case 6: LQ size + scale config
        {
            "name": "LQ Size 64 + Scale 4",
            "config": {
                "datasets": {
                    "train": {
                        "lq_size": 64,
                        "scale": 4,
                        "num_in_ch": 3
                    }
                }
            },
            "expected": (1, 3, 256, 256)
        },
        
        # Case 7: Network scale fallback
        {
            "name": "Network Scale Fallback",
            "config": {
                "network_g": {
                    "scale": 4
                },
                "datasets": {
                    "train": {
                        "num_in_ch": 3
                    }
                }
            },
            "expected": (1, 3, 256, 256)  # 64 * 4 = 256
        },
        
        # Case 8: Non-square crop size
        {
            "name": "Non-Square Crop",
            "config": {
                "datasets": {
                    "train": {
                        "crop_size": [256, 384],
                        "num_in_ch": 3
                    }
                }
            },
            "expected": (1, 3, 256, 384)
        }
    ]
    
    print("Testing Dynamic Input Size Detection")
    print("=" * 60)
    
    passed = 0
    total = len(test_cases)
    
    for case in test_cases:
        result = detect_input_size_from_config(case["config"])
        expected = case["expected"]
        
        status = "PASS" if result == expected else "FAIL"
        print(f"{status} {case['name']}: {result} (expected: {expected})")
        
        if result == expected:
            passed += 1
    
    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = test_input_size_detection()
    
    if success:
        print("\nAll tests passed! Dynamic input size detection working correctly.")
    else:
        print("\nSome tests failed. Check implementation.")
        
    # Demo real-world examples
    print("\n" + "=" * 60)
    print("REAL-WORLD CONFIG EXAMPLES")
    print("=" * 60)
    
    real_configs = [
        {
            "name": "Typical RealESRGAN",
            "config": {
                "datasets": {
                    "train": {
                        "name": "DIV2K",
                        "type": "RealESRGANDataset", 
                        "dataroot_gt": "datasets/DIV2K",
                        "gt_size": 256,
                        "num_in_ch": 3,
                        "use_hflip": True,
                        "use_rot": True
                    }
                },
                "network_g": {
                    "type": "RRDBNet",
                    "scale": 4
                }
            }
        },
        {
            "name": "High-Resolution Training",
            "config": {
                "datasets": {
                    "train": {
                        "gt_size": 512,
                        "num_in_ch": 3
                    }
                }
            }
        },
    ]
    
    for config in real_configs:
        result = detect_input_size_from_config(config["config"])
        print(f"{config['name']}: {result}")
        print(f"  - Batch: {result[0]}")
        print(f"  - Channels: {result[1]}")
        print(f"  - Height: {result[2]}")
        print(f"  - Width: {result[3]}")
        print()