"""训练 2D 潜空间的 AE，运行：python -m legacy.training.trainAE"""

import torch

from common import ROOT, get_device
from legacy.common.data import get_train_loader
from common.tracking import log as wandb_log, wandb_run
from legacy.models.ae import AE

def main() -> None:
    latent_dim = 2
    device = get_device()
    print(f"device:{device}")
    train_loader = get_train_loader(batch_size=64)
    model = AE(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    with wandb_run("mnist-ae", config={"model": "AE", "latent_dim": latent_dim, "epochs": 20, "batch_size": 64, "lr": 1e-3}, tags=["mnist", "ae"]):
        for epoch in range(20):
            total_loss = 0
            for images, _ in train_loader:
                optimizer.zero_grad()
                images = images.to(device)
                reconstructed = model(images)
                loss = criterion(reconstructed, images)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            epoch_loss = total_loss / len(train_loader)
            wandb_log({"epoch": epoch + 1, "train/loss": epoch_loss}, step=epoch + 1)
            print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")

    output_path = ROOT / f"outputs/ae{latent_dim}.pth"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)


if __name__ == "__main__":
    main()
