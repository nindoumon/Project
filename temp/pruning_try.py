import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch_pruning as tp
import os
import copy

# ---------- 1. 定义模型 ----------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
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

# ---------- 4. 统计模型参数和计算量 ----------
def count_params_and_macs(model, example_inputs, device):
    """统计模型的参数量和 MACs"""
    macs, params = tp.utils.count_ops_and_params(model, example_inputs)
    return macs, params

# ---------- 5. 主流程 ----------
def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据
    train_loader, test_loader = load_data()

    # 创建模型
    model = SimpleCNN().to(device)

    # ---------- 第一步：训练基础模型 ----------
    print("\n=== 训练原始模型 ===")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(1, 6):
        train(model, device, train_loader, optimizer, epoch)
    acc_original = test(model, device, test_loader)

    # 统计原始模型大小
    example_inputs = torch.randn(1, 1, 28, 28).to(device)
    orig_macs, orig_params = count_params_and_macs(model, example_inputs, device)
    print(f"原始模型 - 参数量: {orig_params/1e6:.2f}M, MACs: {orig_macs/1e6:.2f}M")

    # 保存原始模型（用于对比）
    torch.save(model.state_dict(), "original_model.pth")
    original_size = os.path.getsize("original_model.pth") / (1024 * 1024)
    print(f"原始模型文件大小: {original_size:.2f} MB")

    # ---------- 第二步：使用 torch-pruning 进行结构化剪枝 ----------
    print("\n=== 开始结构化剪枝 (移除50%的通道) ===")

    # 深拷贝模型，避免影响原始模型
    pruned_model = copy.deepcopy(model)

    # 构建依赖图
    example_inputs = torch.randn(1, 1, 28, 28).to(device)
    DG = tp.DependencyGraph().build_dependency(pruned_model, example_inputs=example_inputs)

    # 定义重要性评估标准：使用 GroupNormImportance（基于 L2 范数）
    imp = tp.importance.GroupMagnitudeImportance(p=2)

    # 忽略最后的全连接层（分类器），不进行剪枝
    ignored_layers = []
    for m in pruned_model.modules():
        if isinstance(m, nn.Linear) and m.out_features == 10:
            ignored_layers.append(m)

    # 初始化剪枝器
    pruner = tp.pruner.MetaPruner(
        model=pruned_model,
        example_inputs=example_inputs,
        importance=imp,
        pruning_ratio=0.5,           # 移除50%的通道
        ignored_layers=ignored_layers,
        global_pruning=True,         # 全局剪枝，自动分配各层的剪枝比例
        round_to=1,                  # 通道数对齐到1的倍数（不强制对齐）
    )

    # 执行剪枝（物理移除通道）
    pruner.step()

    # 统计剪枝后的模型
    pruned_macs, pruned_params = count_params_and_macs(pruned_model, example_inputs, device)
    print(f"剪枝后 - 参数量: {pruned_params/1e6:.2f}M (减少 {100*(1-pruned_params/orig_params):.1f}%)")
    print(f"剪枝后 - MACs: {pruned_macs/1e6:.2f}M (减少 {100*(1-pruned_macs/orig_macs):.1f}%)")

    # 评估剪枝后的模型（未微调）
    acc_pruned = test(pruned_model, device, test_loader)
    print(f"剪枝后（未微调）准确率: {acc_pruned:.2f}%")

    # ---------- 第三步：微调剪枝后的模型 ----------
    print("\n=== 微调剪枝后的模型 ===")
    optimizer = optim.Adam(pruned_model.parameters(), lr=0.001)
    for epoch in range(1, 4):
        train(pruned_model, device, train_loader, optimizer, epoch)
    acc_finetuned = test(pruned_model, device, test_loader)
    print(f"微调后准确率: {acc_finetuned:.2f}%")

    # 保存剪枝并微调后的模型
    torch.save(pruned_model.state_dict(), "pruned_finetuned_model.pth")
    pruned_size = os.path.getsize("pruned_finetuned_model.pth") / (1024 * 1024)
    print(f"剪枝后模型文件大小: {pruned_size:.2f} MB")
    print(f"模型文件压缩比: {original_size / pruned_size:.2f}x")

    # ---------- 第四步：对比结果 ----------
    print("\n=== 最终结果对比 ===")
    print(f"原始模型准确率: {acc_original:.2f}%")
    print(f"剪枝后（未微调）准确率: {acc_pruned:.2f}%")
    print(f"微调后准确率: {acc_finetuned:.2f}%")
    print(f"参数量: {orig_params/1e6:.2f}M → {pruned_params/1e6:.2f}M")
    print(f"MACs: {orig_macs/1e6:.2f}M → {pruned_macs/1e6:.2f}M")
    print(f"模型文件: {original_size:.2f}MB → {pruned_size:.2f}MB")

    # 清理临时文件
    os.remove("original_model.pth")
    os.remove("pruned_finetuned_model.pth")

if __name__ == "__main__":
    main()