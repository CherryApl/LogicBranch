# LogicBranch

Repository for LogicBranch model files and execution scripts.

## 🚀 Getting Started

### Prerequisites
- Python              3.8
- torch               1.9.0+cu111
- torchaudio          0.9.0
- torchvision         0.10.0+cu111
- tqdm                4.67.1

### 🔧 Setup Instructions

1. **Prepare the dataset**:
   ```bash
   # Extract the compressed data file
   unzip ./data/data.rar -d ./data
   ```

2. **Run model evaluation**:
   ```bash
   cd Middle/models
   # This will generate evaluation_results.csv with metrics
   python eval.py

## 📂 Repository Structure

```
LogicBranch/
├── Big/                  # Large-scale test cases
│   └── models/           # Pretrained model weights (.pt files)
├── Middle/
│   └── models/           # Pretrained model weights (.pt files)
├── Mini/
│   └── models/           # Pretrained model weights (.pt files)
├── data/                 # Dataset directory (extract data.rar here)
└── README.md             # This documentation
```
