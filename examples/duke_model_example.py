#Example: Using the DukeMTMC pre-trained model.

import sys
from pathlib import Path

#Add parent directory to path for imports.
sys.path.insert(0, str(Path(__file__).parent.parent))

from NewLUPersons import FeatureExtractor, LUPersonPredictor
from NewLUPersons.utils import set_seed
import torch
import numpy as np


def main():
    #Main example function for DukeMTMC model.
    
    #Set seed for reproducibility.
    set_seed(42)
    
    print("=" * 70)
    print("NewLUPersons: DukeMTMC Pre-trained Model Example")
    print("=" * 70)
    
    #Model path.
    model_path = "duke.pth"
    
    print(f"\n1. Loading DukeMTMC pre-trained model from: {model_path}")
    print("   Note: Download duke.pth from https://github.com/DengpanFu/LUPerson")
    
    try:
        #Initialize feature extractor with DukeMTMC model.
        extractor = FeatureExtractor(
            model_path=model_path,
            device="cuda",
            batch_size=32,
            num_workers=4,
            normalize=True,
        )
        print(f"   [OK] DukeMTMC model loaded successfully")
        print(f"   [OK] Feature dimension: {extractor.feature_dim}")
        print(f"   [OK] Model trained on DukeMTMC dataset")
        print(f"   [OK] Expected performance: mAP=82.27%, CMC@1=90.35%")
    except FileNotFoundError as e:
        print(f"   [FAIL] Error: {e}")
        print(f"   Please download duke.pth from:")
        print(f"   https://github.com/DengpanFu/LUPerson")
        return
    
    #Example 1: Extract features from query images.
    print("\n2. Extracting features from query images")
    print("   (Using dummy images for demonstration)")
    
    try:
        #Create dummy query images (8 persons).
        query_images = torch.randn(8, 3, 224, 224)
        query_features = extractor.extract_features(query_images)
        
        print(f"   [OK] Query features extracted successfully")
        print(f"   [OK] Query features shape: {query_features.shape}")
        print(f"   [OK] Number of query persons: {query_features.shape[0]}")
        print(f"   [OK] Feature dimension: {query_features.shape[1]}")
    except Exception as e:
        print(f"   [FAIL] Error during feature extraction: {e}")
        return
    
    #Example 2: Extract features from gallery images.
    print("\n3. Extracting features from gallery images")
    
    try:
        #Create dummy gallery images (702 persons - DukeMTMC size).
        gallery_images = torch.randn(702, 3, 224, 224)
        gallery_features = extractor.extract_features(gallery_images)
        
        print(f"   [OK] Gallery features extracted successfully")
        print(f"   [OK] Gallery features shape: {gallery_features.shape}")
        print(f"   [OK] Number of gallery persons: {gallery_features.shape[0]}")
    except Exception as e:
        print(f"   [FAIL] Error during gallery feature extraction: {e}")
        return
    
    #Example 3: Use high-level predictor API.
    print("\n4. Using LUPersonPredictor for similarity search")
    
    try:
        #Initialize predictor with DukeMTMC model.
        predictor = LUPersonPredictor(
            model_path=model_path,
            device="cuda",
            batch_size=32,
        )
        
        #Compute similarity between query and gallery.
        similarities = predictor.compute_similarity(
            query_features,
            gallery_features,
            metric="cosine",
        )
        
        print(f"   [OK] Similarity matrix computed")
        print(f"   [OK] Similarity matrix shape: {similarities.shape}")
        print(f"   [OK] Similarity statistics:")
        print(f"      - Mean: {similarities.mean():.6f}")
        print(f"      - Std: {similarities.std():.6f}")
        print(f"      - Min: {similarities.min():.6f}")
        print(f"      - Max: {similarities.max():.6f}")
    except Exception as e:
        print(f"   [FAIL] Error during similarity computation: {e}")
        return
    
    #Example 4: Rank gallery by similarity.
    print("\n5. Ranking gallery by similarity to query")
    
    try:
        #Rank gallery for each query.
        rankings = predictor.rank_by_similarity(
            query_features,
            gallery_features,
            top_k=20,
            metric="cosine",
        )
        
        print(f"   [OK] Rankings computed for {len(rankings)} queries")
        for query_idx in range(min(3, len(rankings))):
            top_indices = rankings[query_idx]
            print(f"      Query {query_idx}: Top-20 gallery indices = {top_indices[:10]}...")
    except Exception as e:
        print(f"   [FAIL] Error during ranking: {e}")
        return
    
    #Example 5: Evaluation metrics.
    print("\n6. Computing evaluation metrics")
    
    try:
        from NewLUPersons.evaluation import compute_rank_metrics
        
        #Create dummy labels for evaluation.
        query_ids = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        gallery_ids = np.arange(702)
        
        #Compute distances (negative similarity for ranking).
        distances = -similarities
        
        #Compute metrics.
        metrics = compute_rank_metrics(
            distances,
            query_ids,
            gallery_ids,
            ranks=[1, 5, 10, 20],
        )
        
        print(f"   [OK] Evaluation metrics computed:")
        for metric_name, metric_value in metrics.items():
            print(f"      - {metric_name}: {metric_value:.4f}")
    except Exception as e:
        print(f"   [FAIL] Error during metric computation: {e}")
        return
    
    #Example 6: Feature statistics.
    print("\n7. Feature statistics and analysis")
    
    try:
        print(f"   [OK] Query features statistics:")
        print(f"      - Mean: {query_features.mean():.6f}")
        print(f"      - Std: {query_features.std():.6f}")
        print(f"      - Min: {query_features.min():.6f}")
        print(f"      - Max: {query_features.max():.6f}")
        print(f"      - L2 norm (first sample): {np.linalg.norm(query_features[0]):.6f}")
        
        print(f"   [OK] Gallery features statistics:")
        print(f"      - Mean: {gallery_features.mean():.6f}")
        print(f"      - Std: {gallery_features.std():.6f}")
        print(f"      - Min: {gallery_features.min():.6f}")
        print(f"      - Max: {gallery_features.max():.6f}")
    except Exception as e:
        print(f"   [FAIL] Error during statistics computation: {e}")
        return
    
    #Example 7: Cross-dataset comparison.
    print("\n8. Cross-dataset comparison note")
    print("   DukeMTMC model characteristics:")
    print("   - Trained on DukeMTMC dataset (702 identities)")
    print("   - Expected mAP: 82.27%")
    print("   - Expected CMC@1: 90.35%")
    print("   - Good for cross-dataset evaluation")
    print("   - Can be used for domain adaptation studies")
    
    print("\n" + "=" * 70)
    print("DukeMTMC Model Example completed successfully!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Download duke.pth from https://github.com/DengpanFu/LUPerson")
    print("2. Place duke.pth in the current directory")
    print("3. Run this example with real images for production use")
    print("4. See examples/market_model_example.py for Market-1501 model usage")
    print("5. Compare results between market.pth and duke.pth models")


if __name__ == "__main__":
    main()

