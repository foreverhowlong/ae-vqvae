import sys
from pathlib import Path
# 获取项目根目录（当前脚本的父目录的父目录）
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "models"))

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ae import AE

latent_dim = 2

device = torch.device('mps' if torch.mps.is_available() else 'cpu')

print(f"device:{device}")

transform = transforms.Compose([
    transforms.ToTensor(),          # PIL图像 → [0,1] 的tensor
    transforms.Normalize((0.5,), (0.5,))  # 归一化到 [-1, 1]
])

train_dataset = datasets.MNIST(root=ROOT / 'data', train=True, 
                                download=True, transform=transform)
test_dataset  = datasets.MNIST(root=ROOT / 'data', train=False, 
                                download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

model = AE(latent_dim=latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()  # 自编码器用 MSE 重建损失

for epoch in range(20):
    total_loss = 0
    for images, _ in train_loader:
        # 清零梯度
        optimizer.zero_grad()
        
        
        images = images.to(device)
        reconstructed = model(images)
        
        # 1. 算 loss（MSE，输入是 reconstructed 和 images）
        loss = criterion(reconstructed,images)
        
        # 3. 反向传播
        loss.backward()
        # 4. 更新参数
        optimizer.step()
        total_loss += loss.item()
        
    print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}')
    
torch.save(model.state_dict(), ROOT / 'outputs/ae2.pth')
