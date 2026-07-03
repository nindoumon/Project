import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.nn.utils import prune
import os
import time

# ---------- 1. 定义模型 ----------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)  # MNIST 为 28x28，经过两次卷积后变为 12x12
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ---------- 2. 数据加载 ----------
def load_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# ---------- 3. 训练/测试函数 ----------
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy

# ---------- 4. 主流程 ----------
def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据
    train_loader, test_loader = load_data()

    # 创建模型
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ---------- 第一步：训练基础模型 ----------
    print("\n=== 训练原始模型 ===")
    for epoch in range(1, 6):  # 5 个 epoch
        train(model, device, train_loader, optimizer, epoch)
    acc_original = test(model, device, test_loader)

    # 保存原始模型的大小
    torch.save(model.state_dict(), "original_model.pth")
    original_size = os.path.getsize("original_model.pth")
    print(f"原始模型大小: {original_size:.2f} MB")

    # ---------- 第二步：全局剪枝（移除 50% 的权重） ----------
    print("\n=== 开始全局剪枝 (移除50%的权重) ===")
    # 收集所有需要剪枝的参数（卷积层和全连接层的权重）
    parameters_to_prune = (
        (model.conv1, 'weight'),
        (model.conv2, 'weight'),
        (model.fc1, 'weight'),
        (model.fc2, 'weight'),
    )
    # 应用全局幅度剪枝
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.5,  # 剪掉 50%
    )

    # 查看剪枝后的稀疏度
    def print_sparsity(model):
        total_params = 0
        zero_params = 0
        for name, param in model.named_parameters():
            if 'weight' in name and 'mask' not in name:  # 只统计原始权重，不统计mask
                total_params += param.numel()
                zero_params += (param == 0).sum().item()
        print(f"总参数: {total_params}, 零参数: {zero_params}, 稀疏度: {100 * zero_params / total_params:.2f}%")
    print("剪枝后的稀疏度:")
    print_sparsity(model)

    # 剪枝后，需要将剪枝永久生效（移除 mask 缓冲，使权重真正变为零）
    # 注意：这一步会永久修改权重，并删除 mask 和 orig 缓冲
    for module, name in parameters_to_prune:
        prune.remove(module, name)  # 这会将 weight 永久设置为剪枝后的权重（含零）

    # 评估剪枝后的模型（未微调）
    acc_pruned = test(model, device, test_loader)
    print(f"剪枝后（未微调）准确率: {acc_pruned:.2f}%")

    # ---------- 第三步：微调剪枝后的模型 ----------
    print("\n=== 微调剪枝后的模型 ===")
    # 重新初始化优化器（因为模型参数变了）
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(1, 4):  # 再训练 3 个 epoch
        train(model, device, train_loader, optimizer, epoch)
    acc_finetuned = test(model, device, test_loader)
    print(f"微调后准确率: {acc_finetuned:.2f}%")

    # 保存剪枝并微调后的模型
    torch.save(model.state_dict(), "pruned_finetuned_model.pth")
    pruned_size = os.path.getsize("pruned_finetuned_model.pth")
    print(f"剪枝后模型大小: {pruned_size:.2f} MB")
    print(f"模型压缩比: {original_size / pruned_size:.2f}x")

    # ---------- 第四步：对比结果 ----------
    print("\n=== 最终结果对比 ===")
    print(f"原始模型准确率: {acc_original:.2f}%")
    print(f"剪枝后（未微调）准确率: {acc_pruned:.2f}%")
    print(f"微调后准确率: {acc_finetuned:.2f}%")
    print(f"模型大小从 {original_size:.2f} MB 减小到 {pruned_size:.2f} MB")

    # 清理临时文件
    os.remove("original_model.pth")
    os.remove("pruned_finetuned_model.pth")

if __name__ == "__main__":
    main()