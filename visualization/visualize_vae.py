import sys
from pathlib import Path
# 获取项目根目录
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "models"))

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from models.vae import VAE

# 要可视化的模型文件路径
model_path = ROOT / "outputs/vae2.pth"


def show_reconstruction(pth_path=model_path, num_pairs=8):
    """加载训练好的 VAE 模型，从测试集取一个 batch 重建，
    并排显示原图和重建图。

    Args:
        pth_path: 模型权重文件路径
        num_pairs: 显示的原图-重建图对数（默认 8）
    """
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    # 加载模型
    model = VAE().to(device)
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.eval()

    # 构建测试集 DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    test_dataset = datasets.MNIST(root=ROOT / 'data', train=False,
                                  download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 取一个 batch
    images, _ = next(iter(test_loader))
    images = images[:num_pairs].to(device)

    with torch.no_grad():
        reconstructed, _, _ = model(images)  # VAE 返回 (x_recon, mu, logvar)

    # 反归一化：[-1, 1] → [0, 1]
    images = (images + 1) / 2
    reconstructed = (reconstructed + 1) / 2

    # 移回 CPU 并裁剪到合法范围
    images = images.cpu().clamp(0, 1)
    reconstructed = reconstructed.cpu().clamp(0, 1)

    fig, axes = plt.subplots(num_pairs, 2, figsize=(4, num_pairs * 2))

    for i in range(num_pairs):
        # 原图
        ax_orig = axes[i, 0]
        ax_orig.imshow(images[i].squeeze(), cmap="gray")
        ax_orig.set_title("Original")
        ax_orig.axis("off")

        # 重建图
        ax_recon = axes[i, 1]
        ax_recon.imshow(reconstructed[i].squeeze(), cmap="gray")
        ax_recon.set_title("Reconstructed")
        ax_recon.axis("off")

    plt.tight_layout()
    plt.show()


def plot_latent_space(pth_path=model_path):
    """加载训练好的 VAE 模型，对测试集所有图像做 encode，
    得到二维 latent vector（mu），按数字类别（0-9）用不同颜色画散点图。

    Args:
        pth_path: 模型权重文件路径
    """
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    # 加载模型
    model = VAE().to(device)
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.eval()

    # 构建测试集 DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    test_dataset = datasets.MNIST(root=ROOT / 'data', train=False,
                                  download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # 对所有测试图像做 encode，收集 latent vectors 和标签
    all_latents = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            mu, _ = model.encoder(images)  # VAE encoder 返回 (mu, logvar)，取 mu 作为 latent
            all_latents.append(mu.cpu())
            all_labels.append(labels)

    all_latents = torch.cat(all_latents, dim=0).numpy()  # (N, 2)
    all_labels = torch.cat(all_labels, dim=0).numpy()     # (N,)

    # 按数字类别用 tab10 颜色画散点图
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(all_latents[:, 0], all_latents[:, 1],
                          c=all_labels, cmap='tab10', s=2, alpha=0.7)
    plt.colorbar(scatter, ticks=range(10), label='Digit class')
    plt.clim(-0.5, 9.5)
    plt.xlabel('Latent dim 1')
    plt.ylabel('Latent dim 2')
    plt.title('VAE Latent Space (2D) — MNIST Test Set')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    show_reconstruction()
    plot_latent_space()