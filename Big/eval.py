import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from dlgn import DiffLogicLayer, GroupSum
import os
import glob
import re
import csv
from datetime import datetime

# 配置参数
config = {
    "batch_size": 128,
    "models_dir": "models",  # 模型目录
    "data_dir": "../data",       # 数据目录
    "results_file": "evaluation_results.csv",  # 结果文件
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

def load_test_data(data_prefix):
    """加载测试数据"""
    sample_path = os.path.join(config['data_dir'], f"{data_prefix}_samples.pt")
    label_path = os.path.join(config['data_dir'], f"{data_prefix}_labels.pt")
    tagep_path = os.path.join(config['data_dir'], f"{data_prefix}_tageps.pt")
    
    samples = torch.load(sample_path)[:, :200]
    labels = torch.load(label_path).long()
    tageps = torch.load(tagep_path).long()
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
    
    return {
        "model_accuracy": acc,
        "tage_accuracy": tage_acc,
        "improvement": acc - tage_acc,
        "test_loss": test_loss,
        "sample_count": len(test_loader.dataset)
    }

def extract_data_prefix(model_path):
    """从模型路径中提取数据前缀"""
    # 从类似 'models/best_model_G_int_0_trace_branch_1_samples.pt.pth' 的路径中提取 'G_int_0_trace_branch_1'
    match = re.search(r'best_model_(.*?)_samples\.pt\.pth', model_path)
    if match:
        return match.group(1)
    return None

def save_results_to_csv(results, filename):
    """保存结果到CSV文件"""
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # 如果文件不存在，写入表头
        if not file_exists:
            writer.writerow([
                'timestamp',
                'model_file',
                'data_prefix',
                'model_accuracy',
                'tage_accuracy',
                'improvement',
                'test_loss',
                'sample_count'
            ])
        
        # 写入数据行
        for result in results:
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                os.path.basename(result['model_path']),
                result['data_prefix'],
                f"{result['model_accuracy']:.2f}",
                f"{result['tage_accuracy']:.2f}",
                f"{result['improvement']:.2f}",
                f"{result['test_loss']:.4f}",
                result['sample_count']
            ])

if __name__ == "__main__":
    # 获取所有模型文件
    model_files = glob.glob(os.path.join(config['models_dir'], 'best_model_*.pth'))
    
    if not model_files:
        print(f"No model files found in {config['models_dir']}")
        exit()
    
    print(f"Found {len(model_files)} model files to evaluate")
    
    results = []
    
    for model_path in model_files:
        # 提取数据前缀
        data_prefix = extract_data_prefix(model_path)
        if not data_prefix:
            print(f"Could not extract data prefix from {model_path}, skipping")
            continue
            
        print(f"\n{'='*50}")
        print(f"Evaluating model: {os.path.basename(model_path)}")
        print(f"Using dataset prefix: {data_prefix}")
        
        try:
            # 1. 加载模型
            model = build_model()
            model.load_state_dict(torch.load(model_path))
            
            # 2. 加载测试数据
            test_dataset = load_test_data(data_prefix)
            test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
            print(f"Loaded {len(test_dataset)} test samples")
            
            # 3. 运行测试
            metrics = evaluate_model(model, test_loader)
            
            # 存储结果
            result = {
                "model_path": model_path,
                "data_prefix": data_prefix,
                **metrics
            }
            results.append(result)
            
            # 打印当前结果
            print(f"\nTest Results for {data_prefix}:")
            print(f"Model Accuracy: {metrics['model_accuracy']:.2f}%")
            print(f"TAGE Accuracy: {metrics['tage_accuracy']:.2f}%")
            print(f"Improvement: {metrics['improvement']:+.2f}%")
            print(f"Test Loss: {metrics['test_loss']:.4f}")
            
            # 立即保存结果到CSV（增量保存）
            save_results_to_csv([result], config['results_file'])
            print(f"Results saved to {config['results_file']}")
            
        except Exception as e:
            print(f"Error evaluating {model_path}: {str(e)}")
            continue
    
    # 打印汇总结果
    print("\n\n" + "="*50)
    print("SUMMARY OF ALL EVALUATIONS")
    print("="*50)
    for result in results:
        print(f"\nModel: {os.path.basename(result['model_path'])}")
        print(f"Dataset: {result['data_prefix']}")
        print(f"Model Accuracy: {result['model_accuracy']:.2f}%")
        print(f"TAGE Accuracy: {result['tage_accuracy']:.2f}%")
        print(f"Improvement: {result['improvement']:+.2f}%")
        print(f"Test Loss: {result['test_loss']:.4f}")
        print(f"Sample Count: {result['sample_count']}")
    
    # 最终保存所有结果到CSV（完整覆盖）
    save_results_to_csv(results, config['results_file'].replace('.csv', '_full.csv'))
    print(f"\nFull results also saved to {config['results_file'].replace('.csv', '_full.csv')}")