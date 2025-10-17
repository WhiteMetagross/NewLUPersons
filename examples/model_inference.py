#!/usr/bin/env python
# encoding: utf-8
"""
Example: Model inference and similarity search.

This example demonstrates how to:
1. Load a pre-trained LUPerson model
2. Perform similarity search
3. Rank persons by similarity
4. Compute evaluation metrics
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from NewLUPersons import LUPersonPredictor
from NewLUPersons.evaluation import compute_rank_metrics
from NewLUPersons.utils import set_seed
import numpy as np
import torch


def main():
    """Main example function."""
    
    # Set seed for reproducibility
    set_seed(42)
    
    print("=" * 60)
    print("NewLUPersons: Model Inference Example")
    print("=" * 60)
    
    # Model path
    model_path = "market.pth"
    
    print(f"\n1. Loading model from: {model_path}")
    
    try:
        # Initialize predictor
        predictor = LUPersonPredictor(
            model_path=model_path,
            device="cuda",
            batch_size=32,
            num_workers=4,
        )
        print(f"   [OK] Model loaded successfully")
    except FileNotFoundError as e:
        print(f"   [FAIL] Error: {e}")
        print(f"   Please ensure {model_path} exists in the current directory")
        return
    
    # Example 1: Extract features
    print("\n2. Extracting features from dummy images")
    
    try:
        # Create dummy query and gallery features
        query_features = torch.randn(5, 3, 224, 224)
        gallery_features = torch.randn(20, 3, 224, 224)
        
        query_feats = predictor.extract_features(query_features)
        gallery_feats = predictor.extract_features(gallery_features)
        
        print(f"   [OK] Query features shape: {query_feats.shape}")
        print(f"   [OK] Gallery features shape: {gallery_feats.shape}")
    except Exception as e:
        print(f"   [FAIL] Error during feature extraction: {e}")
        return
    
    # Example 2: Compute similarity
    print("\n3. Computing similarity between query and gallery")
    
    try:
        similarities = predictor.compute_similarity(
            query_feats,
            gallery_feats,
            metric="cosine",
        )
        print(f"   [OK] Similarity matrix shape: {similarities.shape}")
        print(f"   [OK] Similarity statistics:")
        print(f"      - Mean: {similarities.mean():.6f}")
        print(f"      - Std: {similarities.std():.6f}")
        print(f"      - Min: {similarities.min():.6f}")
        print(f"      - Max: {similarities.max():.6f}")
    except Exception as e:
        print(f"   [FAIL] Error during similarity computation: {e}")
        return
    
    # Example 3: Rank by similarity
    print("\n4. Ranking gallery by similarity to query")
    
    try:
        rankings = predictor.rank_by_similarity(
            query_feats,
            gallery_feats,
            top_k=5,
            metric="cosine",
        )
        
        print(f"   [OK] Rankings computed for {len(rankings)} queries")
        for query_idx, top_indices in rankings.items():
            print(f"      Query {query_idx}: Top-5 gallery indices = {top_indices}")
    except Exception as e:
        print(f"   [FAIL] Error during ranking: {e}")
        return
    
    # Example 4: Evaluation metrics
    print("\n5. Computing evaluation metrics")
    
    try:
        # Create dummy labels for evaluation
        query_ids = np.array([0, 1, 2, 3, 4])
        gallery_ids = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4,
                                5, 5, 6, 6, 7, 7, 8, 8, 9, 9])
        
        # Compute distances (negative similarity for ranking)
        distances = -similarities
        
        # Compute metrics
        metrics = compute_rank_metrics(
            distances,
            query_ids,
            gallery_ids,
            ranks=[1, 5, 10],
        )
        
        print(f"   [OK] Evaluation metrics computed:")
        for metric_name, metric_value in metrics.items():
            print(f"      - {metric_name}: {metric_value:.4f}")
    except Exception as e:
        print(f"   [FAIL] Error during metric computation: {e}")
        return
    
    # Example 5: Similarity search (if we had real images)
    print("\n6. Similarity search example (conceptual)")
    print("   To use with real images:")
    print("   >>> results = predictor.retrieve_similar_persons(")
    print("   ...     query_image='query.jpg',")
    print("   ...     gallery_images=['img1.jpg', 'img2.jpg', ...],")
    print("   ...     top_k=5")
    print("   ... )")
    print("   >>> for result in results:")
    print("   ...     print(f\"{result['image_path']}: {result['similarity']:.4f}\")")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

