import sys
from pathlib import Path
# 获取项目根目录
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "models"))

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from vae import VAE

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

model = VAE(latent_dim=latent_dim).to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


for epoch in range(20):
    total_loss = 0
    for images, _ in train_loader:
        # 清零梯度
        optimizer.zero_grad()
        
        
        x = images.to(device)
        x_recon, mu, logvar = model(x)

        
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        recon_loss = torch.nn.functional.mse_loss(x_recon, x, reduction='sum')
        
        loss = kl_loss + recon_loss
        
        # 3. 反向传播
        loss.backward()
        # 4. 更新参数
        optimizer.step()
        total_loss += loss.item()
        
    print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}')
    
torch.save(model.state_dict(), ROOT / 'outputs/vae2.pth')
