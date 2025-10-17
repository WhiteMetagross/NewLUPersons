# NewLUPersons: Complete CodeBase Index.

## Overview:

This document provides a detailed index of the NewLUPersons library codebase, including all modules, classes, functions, and their purposes.

---

## Package Structure:

```
NewLUPersons/
├── __init__.py              #Main package initialization.and exports
├── config/                  #Configuration management.
│   ├── __init__.py
│   └── defaults.py          #Default configurations.
├── data/                    #Data loading and preprocessing.
│   ├── __init__.py
│   ├── datasets.py          #Dataset utilities.
│   └── transforms.py        #Image transformation pipelines.
├── models/                  #Model building and loading.
│   ├── __init__.py
│   └── builder.py           #Model construction utilities.
├── inference/               #Feature extraction and .prediction
│   ├── __init__.py
│   ├── feature_extractor.py #Feature extraction class.
│   └── predictor.py         #High level prediction API.
├── evaluation/              #Evaluation metrics.
│   ├── __init__.py
│   └── metrics.py           #Evaluation metric functions.
└── utils/                   #Utility functions.
    ├── __init__.py
    └── helpers.py           #Helper functions.
```

---

## Module Reference:

### 1. Main Package (`__init__.py`):

**Purpose:** Main package initialization and public API exports.

**Key Exports:**
- `LUPersonPredictor`: High level API for person re-identification.
- `FeatureExtractor`: Feature extraction class.
- `__version__`: Package version (0.1.0).
- `__author__`: Package author (Mridankan Mandal).

**Usage:**
```python
from NewLUPersons import LUPersonPredictor, FeatureExtractor
```

---

### 2. Configuration Module (`config/`):

**Purpose:** Centralized configuration management for models, data, and inference.

#### `config/__init__.py`
- Exports: `get_default_config()`.
- Returns default configuration dictionary.

#### `config/defaults.py`
- **Classes:**
  - `ModelConfig`: Model loading and inference configuration.
  - `DataConfig`: Data loading and preprocessing configuration.
  - `InferenceConfig`: Inference-specific configuration.

- **Functions:**
  - `get_default_config()`: Returns complete default configuration.

**Usage:**
```python
from NewLUPersons.config import get_default_config
config = get_default_config()
```

---

### 3. Data Module (`data/`):

**Purpose:** Data loading, preprocessing, and transformation utilities.

#### `data/__init__.py`
- Exports: `ImageDataset`, `get_inference_transforms`, `get_training_transforms`.

#### `data/datasets.py`
- **Classes:**
  - `ImageDataset`: Simple dataset for loading images from paths.

- **Methods:**
  - `__init__()`: Initialize dataset with image paths and transforms.
  - `__len__()`: Return dataset size.
  - `__getitem__()`: Get single image item.
  - `from_directory()`: Create dataset from directory.

#### `data/transforms.py`
- **Functions:**
  - `get_inference_transforms()`: Get standard inference transformation pipeline
  - `get_training_transforms()`: Get training transformation pipeline

**Usage:**
```python
from NewLUPersons.data import ImageDataset, get_inference_transforms
transforms = get_inference_transforms()
dataset = ImageDataset(image_paths, transform=transforms)
```

---

### 4. Models Module (`models/`):

**Purpose:** Model building and loading utilities.

#### `models/__init__.py`
- Exports: `build_model()`, `load_model()`

#### `models/builder.py`
- **Functions:**
  - `build_model()`: Build ResNet50 model with specified number of classes.
  - `load_model()`: Load pre-trained model from checkpoint file.
  - `_load_checkpoint()`: Internal function to load checkpoint.

**Usage:**
```python
from NewLUPersons.models import build_model, load_model
model = build_model(num_classes=751)
model = load_model("market.pth")
```

---

### 5. Inference Module (`inference/`):

**Purpose:** Feature extraction and high level prediction API.

#### `inference/__init__.py`
- Exports: `FeatureExtractor`, `LUPersonPredictor`

#### `inference/feature_extractor.py`
- **Classes:**
  - `FeatureExtractor`: Extract deep features from person images.

- **Methods:**
  - `__init__()`: Initialize with model path and configuration.
  - `extract_features()`: Extract features from images (paths or tensors).
  - `extract_features_from_directory()`: Extract features from directory.
  - `_get_feature_dim()`: Infer feature dimension from model.

#### `inference/predictor.py`
- **Classes:**
  - `LUPersonPredictor`: High level API for person re-identification.

- **Methods:**
  - `__init__()`: Initialize predictor with model.
  - `extract_features()`: Extract features from images.
  - `compute_similarity()`: Compute similarity between features.
  - `rank_by_similarity()`: Rank gallery by similarity to query.
  - `retrieve_similar_persons()`: Retrieve similar persons from gallery.

**Usage:**
```python
from NewLUPersons import FeatureExtractor, LUPersonPredictor

#Feature extraction.
extractor = FeatureExtractor("market.pth")
features = extractor.extract_features(images)

#High level API.
predictor = LUPersonPredictor("market.pth")
rankings = predictor.rank_by_similarity(query_features, gallery_features)
```

---

### 6. Evaluation Module (`evaluation/`):

**Purpose:** Evaluation metrics for person re-identification.

#### `evaluation/__init__.py`
- Exports: `compute_cmc()`, `compute_map()`, `compute_rank_metrics()`

#### `evaluation/metrics.py`
- **Functions:**
  - `compute_cmc()`: Compute Cumulative Matching Characteristic curve.
  - `compute_map()`: Compute mean Average Precision.
  - `compute_rank_metrics()`: Compute detailed rank metrics.

**Usage:**
```python
from NewLUPersons.evaluation import compute_cmc, compute_map
cmc = compute_cmc(distances, query_ids, gallery_ids)
mAP = compute_map(distances, query_ids, gallery_ids)
```

---

### 7. Utils Module (`utils/`):

**Purpose:** Utility functions for device management, seeding, and I/O.

#### `utils/__init__.py`
- Exports: `get_device()`, `set_seed()`, `save_features()`, `load_features()`.

#### `utils/helpers.py`
- **Functions:**
  - `get_device()`: Get appropriate device (cuda/cpu).
  - `set_seed()`: Set random seeds for reproducibility.
  - `save_features()`: Save features to NPZ file.
  - `load_features()`: Load features from NPZ file.

**Usage:**
```python
from NewLUPersons.utils import get_device, set_seed, save_features
device = get_device()
set_seed(42)
save_features(features, "output.npz")
```

---

## Key Classes and Methods:

### LUPersonPredictor (Main API):

```python
class LUPersonPredictor:
    def __init__(self, model_path, device=None, batch_size=32, num_workers=4)
    def extract_features(self, images) -> np.ndarray
    def compute_similarity(self, query_features, gallery_features, metric='cosine') -> np.ndarray
    def rank_by_similarity(self, query_features, gallery_features, top_k=10, metric='cosine') -> dict
    def retrieve_similar_persons(self, query_image, gallery_images, top_k=5) -> list
```

### FeatureExtractor:

```python
class FeatureExtractor:
    def __init__(self, model_path, device=None, batch_size=32, num_workers=4, normalize=True)
    def extract_features(self, images) -> np.ndarray
    def extract_features_from_directory(self, directory_path) -> dict
```

---

## Data Flow:

1. **Model Loading**: `load_model()` → Load pre-trained checkpoint
2. **Feature Extraction**: `FeatureExtractor` → Extract 2048-dim features
3. **Similarity Computation**: `compute_similarity()` → Compute distances
4. **Ranking**: `rank_by_similarity()` → Get top-k matches
5. **Evaluation**: `compute_cmc()`, `compute_map()` → Evaluate performance

---

## Dependencies:

**Core:**
- torch, torchvision
- numpy, scipy, scikit-learn

**Image Processing:**
- Pillow, opencv-python

**Configuration:**
- yacs, pyyaml

**Utilities:**
- tqdm, termcolor, tabulate

---

## Testing:

**Test Location:** `tests/test_models.py`

**Test Coverage:**
- Model building and loading.
- Feature extraction.
- Similarity computation.
- Ranking functionality.
- Evaluation metrics.

**Run Tests:**
```bash
pytest tests/ -v
```

---

## References:

- **Original LUPerson**: [https://github.com/DengpanFu/LUPerson](https://github.com/DengpanFu/LUPerson)
- **newFastReID**: [https://github.com/WhiteMetagross/newFastReID](https://github.com/WhiteMetagross/newFastReID)
- **FastReID**: [https://github.com/JDAI-CV/fast-reid](https://github.com/JDAI-CV/fast-reid)

