"""训练 VQ-VAE，运行：python -m training.trainVQVAE"""

import torch

from common import ROOT, get_device
from common.data import get_train_loader
from common.experiment import vq_losses
from models.vqvae import VQVAE

Latent_dim = 8
Codebook_K = 256
Epoches = 50
Beta = 0.2

device = get_device()
print(f"device:{device}")
print(f"latent dim:{Latent_dim}, codebook K:{Codebook_K}, epoches:{Epoches}")

train_loader = get_train_loader(batch_size=64)

model = VQVAE(Latent_dim, Codebook_K).to(device)
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(Epoches):
    all_indices = []
    total_loss = 0
    for images, _ in train_loader:
        optimizer.zero_grad()

        x = images.to(device)
        z_e, z_q_raw, z_q_st, x_recon, indices = model(x)

        all_indices.append(indices)

        loss, _, _, _ = vq_losses(z_e, z_q_raw, x_recon, x, beta=Beta)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}')
    all_indices = torch.cat(all_indices, dim=0)  # 把所有batch拼起来
    used = all_indices.unique().numel()
    print(f"Codebook utilization: {used} / {Codebook_K}")

output_path = ROOT / f"outputs/vqvae{Latent_dim}.pth"
output_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), output_path)
