# NewLUPersons: Installation and Setup Guide.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation Steps](#installation-steps)
3. [Downloading Pre-trained Models](#downloading-pre-trained-models)
4. [Verification](#verification)
5. [Troubleshooting](#troubleshooting)

---

## System Requirements:

### Minimum Requirements:

- **Python**: 3.8 or higher (recommended: 3.11).
- **PyTorch**: 2.0+ with CUDA support (recommended: 2.5+).
- **CUDA**: 11.0+ (recommended: 12.1).
- **cuDNN**: 8.0+ (for CUDA support).
- **RAM**: 8 GB minimum (16 GB recommended).
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended).

### Tested Configuration:

- **Operating System**: Windows 11
- **Python**: 3.11.13
- **PyTorch**: 2.5.1+cu121
- **CUDA**: 12.1
- **cuDNN**: 8.9.1

### Supported Operating Systems:

- Windows 10/11
- Linux (Ubuntu 18.04+, CentOS 7+)
- macOS (Intel and Apple Silicon)

---

## Installation Steps:

### Step 1: Install Python:

Ensure Python 3.8+ is installed:

```bash
python --version
```

If not installed, download from [python.org](https://www.python.org/downloads/)

### Step 2: Install PyTorch with CUDA Support:

Visit [pytorch.org](https://pytorch.org/get-started/locally/) and select your configuration:

**For Windows with CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For Linux with CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CPU-only (not recommended for ReID):**
```bash
pip install torch torchvision torchaudio
```

### Step 3: Clone or Download NewLUPersons:

```bash
# Clone from repository
git clone https://github.com/DengpanFu/LUPerson.git
cd NewLUPersons

# Or download and extract the ZIP file
```

### Step 4: Install NewLUPersons:

Navigate to the NewLUPersons directory and install:

```bash
#Basic installation.
pip install -e .

#With GPU support (FAISS GPU).
pip install -e ".[gpu]"

#With CPU only FAISS.
pip install -e ".[faiss]"

#With all optional dependencies.
pip install -e ".[all]"

#Development installation (includes testing tools).
pip install -e ".[dev]"
```

### Step 5: Verify Installation:

```bash
python -c "from NewLUPersons import FeatureExtractor, LUPersonPredictor; print('Installation successful.')"
```

---

## Downloading Pre-trained Models:

### Model Sources:

Pre-trained models are available from the original LUPerson repository:

**[LUPerson GitHub Repository](https://github.com/DengpanFu/LUPerson)**

### Available Models:

| Model | Dataset | mAP | CMC@1 | File Size | Download |
|-------|---------|-----|-------|-----------|----------|
| ResNet50 | Market-1501 | 91.12 | 96.26 | ~97 MB | `market.pth` |
| ResNet50 | DukeMTMC | 82.27 | 90.35 | ~97 MB | `duke.pth` |

### Download Instructions:

**Option 1: Manual Download:**

1. Visit [https://github.com/DengpanFu/LUPerson](https://github.com/DengpanFu/LUPerson)
2. Navigate to the "Model Zoo" or "Pre-trained Models" section
3. Download `market.pth` and/or `duke.pth`
4. Save to your working directory

**Option 2: Command Line Download:**

```bash
#Download market.pth (Market-1501 model).
wget https://github.com/DengpanFu/LUPerson/releases/download/v1.0/market.pth

#Download duke.pth (DukeMTMC model).
wget https://github.com/DengpanFu/LUPerson/releases/download/v1.0/duke.pth
```

**Option 3: Using Python:**

```python
import urllib.request

#Download market.pth.
url = "https://github.com/DengpanFu/LUPerson/releases/download/v1.0/market.pth"
urllib.request.urlretrieve(url, "market.pth")
print("market.pth downloaded successfully")
```

### Model Placement:

Place downloaded models in one of these locations:

```bash
#Option 1: Current working directory.
./market.pth
./duke.pth

#Option 2: Specific directory.
mkdir -p ./models
cp market.pth ./models/
cp duke.pth ./models/

#Option 3: Use full path in code.
predictor = LUPersonPredictor(model_path="/path/to/market.pth")
```

---

## Verification:

### Test 1: Import Check:

```bash
python -c "from NewLUPersons import FeatureExtractor, LUPersonPredictor; print('Imports OK')"
```

### Test 2: Run Unit Tests:

```bash
pytest tests/ -v
```

Expected output: `8 passed in X.XXs`

### Test 3: Quick Feature Extraction:

```python
from NewLUPersons import FeatureExtractor
import torch

#Initialize extractor.
extractor = FeatureExtractor("market.pth")

#Create dummy image.
dummy_image = torch.randn(1, 3, 224, 224)

#Extract features.
features = extractor.extract_features(dummy_image)
print(f"Features extracted: {features.shape}")
```

---

## Troubleshooting:

### Issue 1: CUDA Not Found:

**Error:** `RuntimeError: CUDA is not available`

**Solution:**
```bash
#Check CUDA installation.
nvidia-smi

#Reinstall PyTorch with correct CUDA version.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Issue 2: Model File Not Found:

**Error:** `FileNotFoundError: market.pth not found`

**Solution:**
```bash
#Verify model file exists.
ls -la market.pth

#Use full path.
from NewLUPersons import FeatureExtractor
extractor = FeatureExtractor("/full/path/to/market.pth")
```

### Issue 3: Out of Memory:

**Error:** `RuntimeError: CUDA out of memory`

**Solution:**
```python
#Reduce batch size.
extractor = FeatureExtractor("market.pth", batch_size=16)

#Or use CPU.
extractor = FeatureExtractor("market.pth", device="cpu")
```

### Issue 4: Import Errors:

**Error:** `ModuleNotFoundError: No module named 'NewLUPersons'`

**Solution:**
```bash
#Reinstall in editable mode.
pip install -e .

#Or add to Python path.
import sys
sys.path.insert(0, '/path/to/NewLUPersons')
```

### Issue 5: PyTorch Version Mismatch:

**Error:** `RuntimeError: Incompatible PyTorch version`

**Solution:**
```bash
#Check PyTorch version.
python -c "import torch; print(torch.__version__)"

#Update PyTorch.
pip install --upgrade torch torchvision
```

---

## Getting Help:

1. **Check Documentation**: Review [README.md](./README.md) and [CodeBaseIndex.md](./CodeBaseIndex.md).
2. **Run Examples**: See [examples/](./examples/) directory
3. **Original Repository**: [LUPerson GitHub](https://github.com/DengpanFu/LUPerson).
4. **Report Issues**: Create an issue on GitHub.

---

## Next Steps:

After successful installation:

1. **Download Models**: Get `market.pth` and `duke.pth` from LUPerson repository.
2. **Run Examples**: Try examples in `examples/` directory.
3. **Read Documentation**: Review [README.md](./README.md) for usage.
4. **Explore API**: Check [CodeBaseIndex.md](./CodeBaseIndex.md) for detailed API reference.

---

## Quick Start:

```python
from NewLUPersons import LUPersonPredictor

#Initialize predictor.
predictor = LUPersonPredictor("market.pth")

#Extract features.
features = predictor.extract_features(images)

#Compute similarity.
similarities = predictor.compute_similarity(query_features, gallery_features)

#Rank by similarity.
rankings = predictor.rank_by_similarity(query_features, gallery_features, top_k=10)
```

---

## Support:

For issues or questions:
- Check [Troubleshooting](#troubleshooting) section.
- Review [CodeBaseIndex.md](./CodeBaseIndex.md).
- Visit [LUPerson GitHub](https://github.com/DengpanFu/LUPerson).

