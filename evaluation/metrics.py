#Evaluation metrics for person re-identification.

from typing import Tuple, List
import numpy as np


#Compute Cumulative Matching Characteristic (CMC) curve.
def compute_cmc(
    distances: np.ndarray,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    query_cameras: np.ndarray = None,
    gallery_cameras: np.ndarray = None,
) -> np.ndarray:
    num_queries = distances.shape[0]
    cmc = np.zeros(distances.shape[1])
    
    for q_idx in range(num_queries):
        #Get ranking.
        ranking = np.argsort(distances[q_idx])
        
        #Check if match.
        query_id = query_ids[q_idx]
        query_cam = query_cameras[q_idx] if query_cameras is not None else None
        
        matches = gallery_ids[ranking] == query_id
        
        #Exclude same camera if provided.
        if query_cam is not None:
            same_camera = gallery_cameras[ranking] == query_cam
            matches = matches & ~same_camera
        
        #Compute CMC.
        cmc += matches
    
    cmc = cmc / num_queries
    cmc = np.cumsum(cmc)
    
    return cmc


#Compute mean Average Precision (mAP).
def compute_map(
    distances: np.ndarray,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    query_cameras: np.ndarray = None,
    gallery_cameras: np.ndarray = None,
) -> float:
    num_queries = distances.shape[0]
    aps = []
    
    for q_idx in range(num_queries):
        #Get ranking.
        ranking = np.argsort(distances[q_idx])
        
        #Check if match.
        query_id = query_ids[q_idx]
        query_cam = query_cameras[q_idx] if query_cameras is not None else None
        
        matches = gallery_ids[ranking] == query_id
        
        #Exclude same camera if provided.
        if query_cam is not None:
            same_camera = gallery_cameras[ranking] == query_cam
            matches = matches & ~same_camera
        
        #Compute AP.
        num_matches = np.sum(matches)
        if num_matches == 0:
            ap = 0.0
        else:
            precisions = np.cumsum(matches) / (np.arange(len(matches)) + 1)
            ap = np.sum(precisions[matches]) / num_matches
        
        aps.append(ap)
    
    return np.mean(aps)


#Compute standard ReID evaluation metrics.
def compute_rank_metrics(
    distances: np.ndarray,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    query_cameras: np.ndarray = None,
    gallery_cameras: np.ndarray = None,
    ranks: List[int] = None,
) -> dict:
    if ranks is None:
        ranks = [1, 5, 10]
    
    #Compute mAP.
    mAP = compute_map(
        distances,
        query_ids,
        gallery_ids,
        query_cameras,
        gallery_cameras,
    )
    
    #Compute CMC.
    cmc = compute_cmc(
        distances,
        query_ids,
        gallery_ids,
        query_cameras,
        gallery_cameras,
    )
    
    #Create results.
    results = {'mAP': mAP}
    for rank in ranks:
        if rank <= len(cmc):
            results[f'CMC@{rank}'] = cmc[rank - 1]
    
    return results
