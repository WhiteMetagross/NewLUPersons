#High level prediction interface for NewLUPersons.

from typing import Optional, List, Union, Dict, Any
import torch
import numpy as np
from scipy.spatial.distance import cdist

from .feature_extractor import FeatureExtractor


class LUPersonPredictor:
    #High level API for person re-identification using LUPerson models.
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        #Initialize LUPersonPredictor with model path and configuration.
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        #Initialize feature extractor.
        self.feature_extractor = FeatureExtractor(
            model_path=model_path,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            normalize=True,
        )
    
    def extract_features(
        self,
        images: Union[str, List[str]],
    ) -> np.ndarray:
        #Extract person features from images.
        return self.feature_extractor.extract_features(images)
    
    def compute_similarity(
        self,
        query_features: np.ndarray,
        gallery_features: np.ndarray,
        metric: str = "cosine",
    ) -> np.ndarray:
        #Compute similarity between query and gallery features.
        distances = cdist(query_features, gallery_features, metric=metric)
        
        #Convert to similarity (for cosine, similarity = 1 - distance).
        if metric == "cosine":
            similarities = 1 - distances
        else:
            #For other metrics, use negative distance as similarity.
            similarities = -distances
        
        return similarities
    
    def rank_by_similarity(
        self,
        query_features: np.ndarray,
        gallery_features: np.ndarray,
        top_k: int = 10,
        metric: str = "cosine",
    ) -> Dict[int, List[int]]:
        #Rank gallery samples by similarity to query samples.
        similarities = self.compute_similarity(
            query_features,
            gallery_features,
            metric=metric,
        )
        
        #Get top-k indices for each query.
        rankings = {}
        for i in range(similarities.shape[0]):
            top_indices = np.argsort(-similarities[i])[:top_k]
            rankings[i] = top_indices.tolist()
        
        return rankings
    
    def retrieve_similar_persons(
        self,
        query_image: str,
        gallery_images: List[str],
        top_k: int = 10,
        metric: str = "cosine",
    ) -> List[Dict[str, Any]]:
        #Retrieve similar persons from gallery given a query image.
        #Extract features.
        query_features = self.extract_features([query_image])
        gallery_features = self.extract_features(gallery_images)
        
        #Compute similarities.
        similarities = self.compute_similarity(
            query_features,
            gallery_features,
            metric=metric,
        )[0]
        
        #Create results.
        results = []
        for idx, sim in enumerate(similarities):
            results.append({
                'image_path': gallery_images[idx],
                'similarity': float(sim),
            })
        
        #Sort by similarity.
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return results[:top_k]
