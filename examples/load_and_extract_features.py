#!/usr/bin/env python
# encoding: utf-8
"""
Example: Load model and extract features from images.

This example demonstrates how to:
1. Load a pre-trained LUPerson model
2. Extract features from images
3. Save and load features
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from NewLUPersons import FeatureExtractor
from NewLUPersons.utils import set_seed, save_features, load_features
import numpy as np


def main():
    """Main example function."""
    
    # Set seed for reproducibility
    set_seed(42)
    
    print("=" * 60)
    print("NewLUPersons: Feature Extraction Example")
    print("=" * 60)
    
    # Model path
    model_path = "market.pth"
    
    print(f"\n1. Loading model from: {model_path}")
    
    try:
        # Initialize feature extractor
        extractor = FeatureExtractor(
            model_path=model_path,
            device="cuda",
            batch_size=32,
            num_workers=4,
            normalize=True,
        )
        print(f"   [OK] Model loaded successfully")
        print(f"   [OK] Feature dimension: {extractor.feature_dim}")
    except FileNotFoundError as e:
        print(f"   [FAIL] Error: {e}")
        print(f"   Please ensure {model_path} exists in the current directory")
        return
    
    # Example 1: Extract features from a single image
    print("\n2. Extracting features from a single image")
    print("   (This is a demonstration - create test images for actual use)")
    
    # For demonstration, we'll create a dummy image tensor
    import torch
    dummy_image = torch.randn(1, 3, 224, 224)
    
    try:
        features = extractor.extract_features(dummy_image)
        print(f"   [OK] Features extracted successfully")
        print(f"   [OK] Feature shape: {features.shape}")
        print(f"   [OK] Feature statistics:")
        print(f"      - Mean: {features.mean():.6f}")
        print(f"      - Std: {features.std():.6f}")
        print(f"      - Min: {features.min():.6f}")
        print(f"      - Max: {features.max():.6f}")
    except Exception as e:
        print(f"   [FAIL] Error during feature extraction: {e}")
        return
    
    # Example 2: Extract features from multiple images
    print("\n3. Extracting features from multiple images")
    
    try:
        # Create dummy batch
        dummy_batch = torch.randn(4, 3, 224, 224)
        features_batch = extractor.extract_features(dummy_batch)
        print(f"   [OK] Batch features extracted successfully")
        print(f"   [OK] Batch feature shape: {features_batch.shape}")
    except Exception as e:
        print(f"   [FAIL] Error during batch feature extraction: {e}")
        return
    
    # Example 3: Save and load features
    print("\n4. Saving and loading features")
    
    try:
        output_path = "example_features.npz"
        metadata = {
            "model": "market.pth",
            "num_images": features_batch.shape[0],
            "feature_dim": features_batch.shape[1],
        }
        
        save_features(features_batch, output_path, metadata=metadata)
        print(f"   [OK] Features saved to: {output_path}")
        
        loaded_features, loaded_metadata = load_features(output_path)
        print(f"   [OK] Features loaded successfully")
        print(f"   [OK] Loaded feature shape: {loaded_features.shape}")
        print(f"   [OK] Loaded metadata: {loaded_metadata}")
        
        # Verify features match
        if np.allclose(features_batch, loaded_features):
            print(f"   [OK] Loaded features match original features")
        else:
            print(f"   [FAIL] Loaded features do not match original features")
    except Exception as e:
        print(f"   [FAIL] Error during save/load: {e}")
        return
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

