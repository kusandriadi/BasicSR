#!/usr/bin/env python3
"""
Test script untuk model stats integration
"""

import sys
import os
sys.path.append('.')

import torch
from basicsr.utils.model_stats import log_discriminator_stats, print_model_lightweight_stats
from basicsr.archs.discriminator_arch import (
    UltraLightDiscriminator, 
    UltraLightDiscriminator_v2, 
    UltraFastQualityDiscriminator
)

def test_model_stats():
    """Test model stats functions"""
    print("Testing Model Stats Functions")
    print("=" * 60)
    
    # Create test models
    models = {
        "UltraLight_v1": UltraLightDiscriminator(num_feat=32),
        "UltraLight_v2": UltraLightDiscriminator_v2(num_feat=32), 
        "UltraFastQuality": UltraFastQualityDiscriminator(num_feat=48)
    }
    
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print('='*60)
        
        # Test basic stats function
        stats = print_model_lightweight_stats(
            model=model,
            model_name=name,
            input_size=(1, 3, 128, 128),
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        print(f"\nReturned stats keys: {list(stats.keys())}")
        
    # Test discriminator-specific logging function
    print(f"\n{'='*60}")
    print("Testing Discriminator-Specific Logging")
    print('='*60)
    
    # Test the specialized discriminator logging function
    test_model = UltraFastQualityDiscriminator(num_feat=48)
    log_discriminator_stats(
        discriminator=test_model,
        discriminator_name="UltraFastQuality_Test",
        input_size=(1, 3, 128, 128)
    )

if __name__ == "__main__":
    test_model_stats()