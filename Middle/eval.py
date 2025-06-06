import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from dlgn import DiffLogicLayer, GroupSum
import os

# 配置参数
config = {
    "batch_size": 128,
    "model_path": "models/best_model_G_int_0_trace_branch_1_samples.pt.pth",  # 替换为你的模型路径
    "test_sample_path": "v3/G_int_0_trace_branch_1_samples.pt",  # 测试数据路径
    "test_label_path": "v3/G_int_0_trace_branch_1_labels.pt",    # 测试标签路径
    "test_tagep_path": "v3/G_int_0_trace_branch_1_tageps.pt",    # TAGE预测标签路径
}

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_model():
    """创建新的模型实例"""
    return nn.Sequential(
        nn.Flatten(),
        DiffLogicLayer(200, 400, 1, 4),
        DiffLogicLayer(400, 8000, 2, 2),
        DiffLogicLayer(8000, 4000, 3, 2),
        DiffLogicLayer(4000, 2000, 3, 2),
        DiffLogicLayer(2000, 1000, 4, 4),
        GroupSum(k=2, tau=30)
    ).to(device)

def load_test_data():
    """加载测试数据"""
    samples = torch.load(config['test_sample_path'])[:, :200]
    labels = torch.load(config['test_label_path']).long()
    tageps = torch.load(config['test_tagep_path']).long()
    return TensorDataset(samples, labels, tageps)

def evaluate_model(model, test_loader):
    """评估模型性能"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    tage_correct = 0
    
    with torch.no_grad():
        for data, target, tagep in tqdm(test_loader, desc="Testing"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            test_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            tage_correct += tagep.eq(target.cpu()).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    tage_acc = 100. * tage_correct / len(test_loader.dataset)
    
    print(f"\nTest Results:")
    print(f"Model Accuracy: {acc:.2f}%")
    print(f"TAGE Accuracy: {tage_acc:.2f}%")
    print(f"Improvement: {acc - tage_acc:+.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    # 1. 加载模型
    model = build_model()
    model.load_state_dict(torch.load(config['model_path']))
    print(f"Model loaded from {config['model_path']}")

    # 2. 加载测试数据
    test_dataset = load_test_data()
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    print(f"Loaded {len(test_dataset)} test samples")

    # 3. 运行测试
    evaluate_model(model, test_loader)