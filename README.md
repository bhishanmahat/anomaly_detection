# MPDD Anomaly Detection with GAS

Implementation of Global Anomaly Synthesis (GAS) for metal parts defect detection using the MPDD dataset.

## 🎯 Project Overview

This project implements an anomaly detection system for industrial quality control, specifically targeting metal parts defect detection. The implementation uses Global Anomaly Synthesis (GAS) with manifold hypothesis to distinguish between good and defective parts.

## 📊 Dataset

**MPDD (Metal Parts Defect Detection)**: https://drive.google.com/file/d/1b3dcRqTXR7LZkOEkVQ9qO_EcKzzC2EEI/view
- **Categories**: Bracket (brown, white, black), Metal plate
- **Current Implementation**: Supports all categories
- **Structure**: Train/Test splits with good/bad part classification

## 🏗️ Architecture

### Core Components
1. **Feature Extractor**: WideResNet50 backbone (frozen)
2. **Feature Adaptor**: Linear layer for domain adaptation  
3. **Discriminator**: MLP for anomaly classification
4. **GAS Module**: Global Anomaly Synthesis with manifold hypothesis

## 🚀 How to Run

### Prerequisites
```bash
# Create & activate a virtualenv
conda create -n env_anom-detect python=3.13
conda activate anom_detect
```

### Installation
```bash
# Clone the repo
git clone https://github.com/bhishanmahat/anomaly_detection.git
cd anomaly_detection

# Install CUDA‑enabled PyTorch (for NVIDIA GPUs)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install the packages
pip install -r requirements.txt

```

### Dataset Setup
1. Download MPDD dataset from: https://drive.google.com/file/d/1b3dcRqTXR7LZkOEkVQ9qO_EcKzzC2EEI/view
2. In your project root, create `data/`
3. Unzip the dataset into `data/` to match the following structure:
```
├── data/
    ├── anomaly_dataset/
        ├── bracket_white/
        │   ├── ground_truth/
        |   |   ├── defective_painting/
        |   |   ├── scratches/ 
        │   ├── test
        |   |   ├── defective_painting/
        |   |   ├── good/
        |   |   ├── scratches/ 
        │   └── train/good/
        ├── bracket_brown/
        ├── bracket_black/
        └── metal_plate/
```

### Training
```bash
python scripts/main.py
```

### Configuration
Modify settings in `main.py`:
```python
DATASET_PATH = "./data/anomaly_dataset"
CATEGORY = "bracket_white"  # Choose: bracket_white, bracket_brown, bracket_black, metal_plate
RESULTS_DIR = "./results"
BATCH_SIZE = 8
IMAGE_SIZE = 288
EPOCHS = 100
```

## 📁 File Structure

```
├── data/anomaly_dataset/
├── scripts/
|   ├── main.py              # Entry point and configuration
|   ├── trainer.py           # Training loop and model management
|   ├── gas.py               # Global Anomaly Synthesis implementation
|   ├── models.py            # Neural network architectures
|   ├── datloader.py          # Data loading and preprocessing
|   ├── visualize.py         # Plotting and visualization
├── results/                 # Output directory for models and plots
```

## 🔧 Model Details

### Hyperparameters
- **Learning Rate**: 0.0002 (Adam optimizer)
- **GAS Parameters**:
  - `r1 = 1.0`, `r2 = 2.0` (manifold constraints)
  - `eta = 0.1` (gradient ascent learning rate)
  - `n_step = 5`, `n_proj = 1` (iteration parameters)
- **Image Size**: 288×288
- **Feature Dimensions**: 1536 (WideResNet50 layer2 + layer3)

## 🔬 Technical Implementation

### Global Anomaly Synthesis (GAS)
1. **Gaussian Noise**: Add random noise to normal features
2. **Gradient Ascent**: Move features toward anomaly direction  
3. **Truncated Projection**: Constrain within manifold bounds

### Key Features
- **Best Model Saving**: Automatically saves highest AUC checkpoint
- **Individual Loss Tracking**: Monitor normal vs anomaly loss separately
- **Image-level Evaluation**: Spatial max pooling for final scores
- **Memory Efficient**: Optimized tensor operations

## 📈 Output

After training, the following files will be generated in `results/`:
- `training_curves_{category}.png` - Loss and AUC plots
- `roc_curve_{category}.png` - ROC curve with AUC score
- `best_model_{category}.pth` - Best performing model checkpoint

## 📚 References

- **GLASS Paper**: "A Unified Anomaly Synthesis Strategy with Gradient Ascent for Industrial Anomaly Detection and Localization" - https://arxiv.org/abs/2407.09359
- **MPDD Dataset**: https://github.com/stepanje/MPDD
- **WideResNet**: "Wide Residual Networks" (Zagoruyko & Komodakis, 2016)
