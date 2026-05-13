import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from vqvae import VQVAE
import torch.nn.functional as F

latent_dim = 2
codebook_K = 256

device = torch.device('mps' if torch.mps.is_available() else 'cpu')

print(f"device:{device}")

transform = transforms.Compose([
    transforms.ToTensor(),          # PIL图像 → [0,1] 的tensor
    transforms.Normalize((0.5,), (0.5,))  # 归一化到 [-1, 1]
])

train_dataset = datasets.MNIST(root='./data', train=True, 
                                download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, 
                                download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

model = VQVAE(latent_dim,codebook_K).to(device)


optimizer = torch.optim.Adam(model.parameters())

for epoch in range(20):
    all_indices = []   
    total_loss = 0
    for images, _ in train_loader:
        # 清零梯度
        optimizer.zero_grad()
        
        
        x = images.to(device)
        z_e, z_q_raw,z_q_st, x_recon, indices = model(x)

        all_indices.append(indices)
        
        recon_loss = F.mse_loss(x_recon, x)
        codebook_loss = F.mse_loss(z_q_raw,z_e.detach())
        commitment_loss = F.mse_loss(z_e,z_q_raw.detach())
        
        beta = 0.2
        
        loss = codebook_loss + beta * commitment_loss + recon_loss
        
        
        # 3. 反向传播
        loss.backward()
        # 4. 更新参数
        optimizer.step()
        total_loss += loss.item()
        
    print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}')
    all_indices = torch.cat(all_indices, dim=0)  # 把所有batch拼起来
    used = all_indices.unique().numel()
    print(f"Codebook utilization: {used} / {codebook_K}")
    
torch.save(model.state_dict(), 'outputs/vqvae2.pth')
