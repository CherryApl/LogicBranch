# LogicBranch

Repository for LogicBranch model files and execution scripts.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 1.10+
- Other dependencies: `pip install -r requirements.txt`
- Python              3.8
- torch               1.9.0+cu111
- torchaudio          0.9.0
- torchvision         0.10.0+cu111
- tqdm                4.67.1

### 🔧 Setup Instructions

1. **Prepare the dataset**:
   ```bash
   # Extract the compressed data file
   unzip ./Middle/data/data.rar -d ./Middle/data/

2. **Run model evaluation**:
```bash
  cd Middle/models
  # This will generate evaluation_results.csv with metrics
  python eval.py

