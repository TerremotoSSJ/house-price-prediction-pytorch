# house-price-prediction-pytorch
## What the project does?

This project implements a neural network for predicting California housing prices using the "California Housing Prices" dataset from Kaggle. The model incorporates Dropout Regularization techniques to prevent overfitting and improve generalization.

**Used Dataset**: https://www.kaggle.com/datasets/camnugent/california-housing-prices

## Features:

**Neural Network Architecture**: Multi-layer perceptron with dropout layers

**Regularization**: Dropout technique to prevent overfitting

**Data Preprocessing**: One-hot encoding, standardization, and missing value handling

**Training Pipeline**: Complete training and evaluation workflow

**Model Persistence**: Save and load trained models for inference and testing

## Dataset

**Source**: California Housing Prices on Kaggle

**Features**: 9 attributes including geographical, demographic, and housing characteristics

**Target**: Median house value for California districts

**Samples**: 20,640 entries

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step-by-Step Installation

1. **Clone the repository**
```bash
git clone https://github.com/TerremotoSSJ/house-price-prediction-pytorch.git
cd california-housing-prediction
```

### Install core dependencies (will auto-select appropriate torch version)
```bash
pip install -r requirements.txt
```
### For CPU-only systems:
```bash
pip install torch==2.10.0+cpu pandas==2.3.3 scikit-learn==1.8.0
```
### For GPU (CUDA 12.1):
```bash
pip install torch==2.10.0+cu121 pandas==2.3.3 scikit-learn==1.8.0
```

### Verify Installation
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import pandas; print(f'Pandas version: {pandas.__version__}')"
python -c "import sklearn; print(f'Scikit-learn version: {sklearn.__version__}')"
```

## Structure
```bash
project/
├── data/              # Datasets
├── src/               # 
│   ├── dataset.py     # Dataset class
│   ├── eval.py        # Evaluating
│   ├── model.py       # Neural Network
│   └── train.py       # Training
├── experiments/       # test scripts
├── requirements.txt  # Dependecies
├── models/           # Save models
│   ├── Default.pth   # Default model
│   .
│   .                 #Save all the models is needed want to save
│   .
│   └── LastModel.pth 
└── README.md        # Documentation
```