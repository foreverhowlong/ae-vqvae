import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from common import ROOT, get_device
from legacy.common.data import get_test_loader
from legacy.models.vae import VAE
from legacy.models.vqvae import VQVAE

# 模型文件路径
vae_path = ROOT / "outputs/vae32.pth"
vqvae_path = ROOT / "outputs/vqvae32.pth"

# 模型参数（与训练时一致）
latent_dim = 32
codebook_K = 256


def get_test_data(batch_size=64):
    return get_test_loader(batch_size=batch_size)


def show_reconstruction_comparison(num_pairs=8):
    """加载训练好的 VAE 和 VQ-VAE 模型，从测试集取一个 batch，
    对比显示原图、VAE 重建图、VQ-VAE 重建图。

    Args:
        num_pairs: 显示的样本数（默认 8）
    """
    device = get_device()

    # ---- 加载模型 ----
    vae_model = VAE(latent_dim=latent_dim).to(device)
    vae_model.load_state_dict(torch.load(vae_path, map_location=device))
    vae_model.eval()

    vqvae_model = VQVAE(latent_dim=latent_dim, codebook_K=codebook_K).to(device)
    vqvae_model.load_state_dict(torch.load(vqvae_path, map_location=device, weights_only=True))
    vqvae_model.eval()

    # ---- 获取测试数据 ----
    test_loader = get_test_data(batch_size=64)
    images, labels = next(iter(test_loader))
    images = images[:num_pairs].to(device)
    labels = labels[:num_pairs]

    # ---- 推理 ----
    with torch.no_grad():
        vae_recon, _, _ = vae_model(images)          # VAE: (x_recon, mu, logvar)
        _, _, _, vqvae_recon, _ = vqvae_model(images)  # VQ-VAE: (z_e, z_q_raw, z_q_st, x_recon, indices)

    # ---- 反归一化：[-1, 1] → [0, 1] ----
    images = ((images + 1) / 2).cpu().clamp(0, 1)
    vae_recon = ((vae_recon + 1) / 2).cpu().clamp(0, 1)
    vqvae_recon = ((vqvae_recon + 1) / 2).cpu().clamp(0, 1)

    # ---- 计算 MSE 和 PSNR ----
    mse_vae = ((images - vae_recon) ** 2).view(num_pairs, -1).mean(dim=1)
    mse_vqvae = ((images - vqvae_recon) ** 2).view(num_pairs, -1).mean(dim=1)
    psnr_vae = 20 * torch.log10(1.0 / torch.sqrt(mse_vae))
    psnr_vqvae = 20 * torch.log10(1.0 / torch.sqrt(mse_vqvae))

    avg_mse_vae = mse_vae.mean().item()
    avg_mse_vqvae = mse_vqvae.mean().item()
    avg_psnr_vae = psnr_vae.mean().item()
    avg_psnr_vqvae = psnr_vqvae.mean().item()

    # ---- 可视化：每行 3 列（原图 | VAE 重建 | VQ-VAE 重建） ----
    fig, axes = plt.subplots(num_pairs, 3, figsize=(9, num_pairs * 2.2))

    for i in range(num_pairs):
        # 原图
        ax_orig = axes[i, 0]
        ax_orig.imshow(images[i].squeeze(), cmap="gray")
        ax_orig.set_title(f"Original\n(label={labels[i].item()})", fontsize=9)
        ax_orig.axis("off")

        # VAE 重建
        ax_vae = axes[i, 1]
        ax_vae.imshow(vae_recon[i].squeeze(), cmap="gray")
        ax_vae.set_title(f"VAE\nPSNR={psnr_vae[i]:.1f}dB", fontsize=9, color="blue")
        ax_vae.axis("off")

        # VQ-VAE 重建
        ax_vqvae = axes[i, 2]
        ax_vqvae.imshow(vqvae_recon[i].squeeze(), cmap="gray")
        ax_vqvae.set_title(f"VQ-VAE\nPSNR={psnr_vqvae[i]:.1f}dB", fontsize=9, color="red")
        ax_vqvae.axis("off")

    # 顶部标题
    fig.suptitle(
        f"VAE vs VQ-VAE Reconstruction Comparison\n"
        f"Avg MSE — VAE: {avg_mse_vae:.6f}, VQ-VAE: {avg_mse_vqvae:.6f}  |  "
        f"Avg PSNR — VAE: {avg_psnr_vae:.2f}dB, VQ-VAE: {avg_psnr_vqvae:.2f}dB",
        fontsize=12, fontweight="bold"
    )

    plt.tight_layout()
    plt.show()


def show_latent_space_comparison():
    """对比 VAE 和 VQ-VAE 的 latent space 散点图。
    VAE: 使用 encoder 输出的 mu 作为 latent vector
    VQ-VAE: 使用 encoder 输出的 z_e 作为 latent vector
    """
    device = get_device()

    # ---- 加载模型 ----
    vae_model = VAE(latent_dim=latent_dim).to(device)
    vae_model.load_state_dict(torch.load(vae_path, map_location=device))
    vae_model.eval()

    vqvae_model = VQVAE(latent_dim=latent_dim, codebook_K=codebook_K).to(device)
    vqvae_model.load_state_dict(torch.load(vqvae_path, map_location=device, weights_only=True))
    vqvae_model.eval()

    # ---- 获取测试数据 ----
    test_loader = get_test_data(batch_size=256)

    # ---- 收集 VAE latent vectors ----
    vae_latents, vae_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            mu, _ = vae_model.encoder(images)
            vae_latents.append(mu.cpu())
            vae_labels.append(labels)

    vae_latents = torch.cat(vae_latents, dim=0).numpy()
    vae_labels = torch.cat(vae_labels, dim=0).numpy()

    # ---- 收集 VQ-VAE latent vectors (z_e) ----
    vqvae_latents, vqvae_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            z_e = vqvae_model.encoder(images)
            vqvae_latents.append(z_e.cpu())
            vqvae_labels.append(labels)

    vqvae_latents = torch.cat(vqvae_latents, dim=0).numpy()
    vqvae_labels = torch.cat(vqvae_labels, dim=0).numpy()

    # ---- 对两个模型的 latent 都做 PCA 降维到 2 维以便可视化 ----
    pca_vae = PCA(n_components=2)
    vae_latents_2d = pca_vae.fit_transform(vae_latents)
    vae_explained = pca_vae.explained_variance_ratio_.sum()

    pca_vqvae = PCA(n_components=2)
    vqvae_latents_2d = pca_vqvae.fit_transform(vqvae_latents)
    vqvae_explained = pca_vqvae.explained_variance_ratio_.sum()

    # ---- 绘制对比图 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # VAE latent space (PCA 降维后)
    scatter1 = ax1.scatter(
        vae_latents_2d[:, 0], vae_latents_2d[:, 1],
        c=vae_labels, cmap='tab10', s=2, alpha=0.7
    )
    plt.colorbar(scatter1, ax=ax1, ticks=range(10), label='Digit class')
    scatter1.set_clim(-0.5, 9.5)
    ax1.set_xlabel('PC 1')
    ax1.set_ylabel('PC 2')
    ax1.set_title(f'VAE Latent Space (mu → PCA 2D)\n'
                  f'Explained variance: {vae_explained:.1%}', fontsize=11)

    # VQ-VAE latent space (PCA 降维后)
    scatter2 = ax2.scatter(
        vqvae_latents_2d[:, 0], vqvae_latents_2d[:, 1],
        c=vqvae_labels, cmap='tab10', s=2, alpha=0.7
    )
    plt.colorbar(scatter2, ax=ax2, ticks=range(10), label='Digit class')
    scatter2.set_clim(-0.5, 9.5)
    ax2.set_xlabel('PC 1')
    ax2.set_ylabel('PC 2')
    ax2.set_title(f'VQ-VAE Latent Space (z_e → PCA 2D)\n'
                  f'Explained variance: {vqvae_explained:.1%}', fontsize=11)

    plt.suptitle('Latent Space Comparison: VAE vs VQ-VAE (PCA 2D) — MNIST Test Set',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 对比 1：重建质量对比（原图 | VAE | VQ-VAE）
    show_reconstruction_comparison(num_pairs=8)

    # 对比 2：Latent space 散点图对比
    show_latent_space_comparison()
