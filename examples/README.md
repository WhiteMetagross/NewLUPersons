# NewLUPersons Examples

This directory contains comprehensive examples demonstrating how to use the NewLUPersons library for person re-identification tasks.

---

## Available Examples

### 1. Feature Extraction (`load_and_extract_features.py`)

**Purpose:** Demonstrates basic feature extraction from images.

**What it covers:**
- Loading a pre-trained model
- Extracting features from single images
- Extracting features from image batches
- Saving and loading features to/from disk
- Feature statistics and analysis

**Usage:**
```bash
python examples/load_and_extract_features.py
```

**Key Functions:**
```python
from NewLUPersons import FeatureExtractor

extractor = FeatureExtractor("market.pth")
features = extractor.extract_features(images)
```

---

### 2. Model Inference (`model_inference.py`)

**Purpose:** Demonstrates inference and similarity search operations.

**What it covers:**
- Loading a pre-trained model
- Extracting features from query and gallery images
- Computing similarity between features
- Ranking gallery by similarity
- Computing evaluation metrics (CMC, mAP)

**Usage:**
```bash
python examples/model_inference.py
```

**Key Functions:**
```python
from NewLUPersons import LUPersonPredictor

predictor = LUPersonPredictor("market.pth")
similarities = predictor.compute_similarity(query_features, gallery_features)
rankings = predictor.rank_by_similarity(query_features, gallery_features, top_k=10)
```

---

### 3. Market-1501 Model (`market_model_example.py`)

**Purpose:** Demonstrates using the Market-1501 pre-trained model.

**What it covers:**
- Loading Market-1501 pre-trained model
- Feature extraction from query and gallery images
- Similarity computation and ranking
- Evaluation metrics computation
- Feature statistics and analysis

**Model Information:**
- **Dataset:** Market-1501 (1,501 identities)
- **Performance:** mAP=91.12%, CMC@1=96.26%
- **File:** `market.pth` (~97 MB)

**Usage:**
```bash
python examples/market_model_example.py
```

**Download Model:**
```bash
# From LUPerson GitHub repository
wget https://github.com/DengpanFu/LUPerson/releases/download/v1.0/market.pth
```

---

### 4. DukeMTMC Model (`duke_model_example.py`)

**Purpose:** Demonstrates using the DukeMTMC pre-trained model.

**What it covers:**
- Loading DukeMTMC pre-trained model
- Feature extraction from query and gallery images
- Similarity computation and ranking
- Evaluation metrics computation
- Cross-dataset comparison notes

**Model Information:**
- **Dataset:** DukeMTMC (702 identities)
- **Performance:** mAP=82.27%, CMC@1=90.35%
- **File:** `duke.pth` (~97 MB)

**Usage:**
```bash
python examples/duke_model_example.py
```

**Download Model:**
```bash
# From LUPerson GitHub repository
wget https://github.com/DengpanFu/LUPerson/releases/download/v1.0/duke.pth
```

---

## Getting Started

### Step 1: Install NewLUPersons

```bash
pip install -e .
```

### Step 2: Download Pre-trained Models

Visit [LUPerson GitHub](https://github.com/DengpanFu/LUPerson) and download:
- `market.pth` (Market-1501 model)
- `duke.pth` (DukeMTMC model)

Place them in the current directory or specify the path in your code.

### Step 3: Run Examples

```bash
# Run feature extraction example
python examples/load_and_extract_features.py

# Run inference example
python examples/model_inference.py

# Run Market-1501 model example
python examples/market_model_example.py

# Run DukeMTMC model example
python examples/duke_model_example.py
```

---

## Quick Start Code

### Basic Feature Extraction

```python
from NewLUPersons import FeatureExtractor
import torch

# Initialize extractor
extractor = FeatureExtractor("market.pth")

# Create dummy images (batch_size=4, channels=3, height=224, width=224)
images = torch.randn(4, 3, 224, 224)

# Extract features
features = extractor.extract_features(images)
print(f"Features shape: {features.shape}")  # (4, 2048)
```

### Similarity Search

```python
from NewLUPersons import LUPersonPredictor
import torch

# Initialize predictor
predictor = LUPersonPredictor("market.pth")

# Create dummy query and gallery images
query_images = torch.randn(5, 3, 224, 224)
gallery_images = torch.randn(100, 3, 224, 224)

# Extract features
query_features = predictor.extract_features(query_images)
gallery_features = predictor.extract_features(gallery_images)

# Compute similarity
similarities = predictor.compute_similarity(query_features, gallery_features)

# Rank by similarity
rankings = predictor.rank_by_similarity(query_features, gallery_features, top_k=10)

# Print results
for query_idx, top_indices in rankings.items():
    print(f"Query {query_idx}: Top-10 gallery indices = {top_indices}")
```

### Evaluation Metrics

```python
from NewLUPersons.evaluation import compute_rank_metrics
import numpy as np

# Create dummy labels
query_ids = np.array([0, 1, 2, 3, 4])
gallery_ids = np.arange(100)

# Compute distances (negative similarity)
distances = -similarities

# Compute metrics
metrics = compute_rank_metrics(distances, query_ids, gallery_ids, ranks=[1, 5, 10])

# Print metrics
for metric_name, metric_value in metrics.items():
    print(f"{metric_name}: {metric_value:.4f}")
```

---

## Model Comparison

| Aspect | Market-1501 | DukeMTMC |
|--------|-------------|----------|
| **Dataset** | Market-1501 | DukeMTMC |
| **Identities** | 1,501 | 702 |
| **Images** | 32,668 | 36,411 |
| **mAP** | 91.12% | 82.27% |
| **CMC@1** | 96.26% | 90.35% |
| **File** | market.pth | duke.pth |
| **Use Case** | General ReID | Cross-dataset eval |

---

## Common Tasks

### Task 1: Extract Features from Directory

```python
from NewLUPersons import FeatureExtractor

extractor = FeatureExtractor("market.pth")
features_dict = extractor.extract_features_from_directory("path/to/images")

for filename, features in features_dict.items():
    print(f"{filename}: {features.shape}")
```

### Task 2: Save and Load Features

```python
from NewLUPersons.utils import save_features, load_features

# Save features
save_features(features, "output.npz", metadata={"model": "market.pth"})

# Load features
loaded_features, metadata = load_features("output.npz")
```

### Task 3: Set Random Seed

```python
from NewLUPersons.utils import set_seed

# Set seed for reproducibility
set_seed(42)
```

### Task 4: Get Device

```python
from NewLUPersons.utils import get_device

# Get appropriate device (cuda or cpu)
device = get_device()
print(f"Using device: {device}")
```

---

## Troubleshooting

### Issue: Model file not found

**Solution:** Download the model from [LUPerson GitHub](https://github.com/DengpanFu/LUPerson) and place it in the current directory.

### Issue: CUDA out of memory

**Solution:** Reduce batch size:
```python
extractor = FeatureExtractor("market.pth", batch_size=16)
```

### Issue: Import errors

**Solution:** Ensure NewLUPersons is installed:
```bash
pip install -e .
```

---

## Next Steps

1. **Explore the API:** Check [CodeBaseIndex.md](../CodeBaseIndex.md) for detailed API reference
2. **Read Documentation:** Review [README.md](../README.md) for comprehensive guide
3. **Installation Help:** See [InstallationAndSetup.md](../InstallationAndSetup.md) for setup instructions
4. **Original Repository:** Visit [LUPerson GitHub](https://github.com/DengpanFu/LUPerson) for more information

---

## References

- **LUPerson Paper:** [Unsupervised Pre-training for Person Re-identification](https://arxiv.org/abs/2104.14294)
- **Original Repository:** [https://github.com/DengpanFu/LUPerson](https://github.com/DengpanFu/LUPerson)
- **FastReID:** [https://github.com/JDAI-CV/fast-reid](https://github.com/JDAI-CV/fast-reid)

