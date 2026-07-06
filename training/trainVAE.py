"""训练 2D 潜空间的 VAE，运行：python -m training.trainVAE"""

import torch
import torch.nn.functional as F

from common import ROOT, get_device
from common.data import get_train_loader
from models.vae import VAE

latent_dim = 2

device = get_device()
print(f"device:{device}")

train_loader = get_train_loader(batch_size=64)

model = VAE(latent_dim=latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(20):
    total_loss = 0
    for images, _ in train_loader:
        optimizer.zero_grad()

        x = images.to(device)
        x_recon, mu, logvar = model(x)

        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')

        loss = kl_loss + recon_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}')

output_path = ROOT / f"outputs/vae{latent_dim}.pth"
output_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), output_path)
