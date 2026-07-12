"""训练 2D 潜空间的 VAE，运行：python -m legacy.training.trainVAE"""

import torch
import torch.nn.functional as F

from common import ROOT, get_device
from legacy.common.data import get_train_loader
from common.tracking import log as wandb_log, wandb_run
from legacy.models.vae import VAE

def main() -> None:
    latent_dim = 2
    device = get_device()
    print(f"device:{device}")
    train_loader = get_train_loader(batch_size=64)
    model = VAE(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    with wandb_run("mnist-vae", config={"model": "VAE", "latent_dim": latent_dim, "epochs": 20, "batch_size": 64, "lr": 1e-3}, tags=["mnist", "vae"]):
        for epoch in range(20):
            total_loss = 0
            for images, _ in train_loader:
                optimizer.zero_grad()
                x = images.to(device)
                x_recon, mu, logvar = model(x)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                recon_loss = F.mse_loss(x_recon, x, reduction="sum")
                loss = kl_loss + recon_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            epoch_loss = total_loss / len(train_loader)
            wandb_log({"epoch": epoch + 1, "train/loss": epoch_loss}, step=epoch + 1)
            print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")

    output_path = ROOT / f"outputs/vae{latent_dim}.pth"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)


if __name__ == "__main__":
    main()
