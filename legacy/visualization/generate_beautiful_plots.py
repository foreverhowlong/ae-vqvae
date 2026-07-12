import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from common import ROOT, get_device
from legacy.common.data import get_test_loader
from legacy.models.vqvae import VQVAE

# 定义输出路径
OUTPUT_DIR = ROOT / "outputs" / "beautiful_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 模型路径
model2_path = ROOT / "outputs/vqvae2.pth"
model32_path = ROOT / "outputs/vqvae32.pth"

def load_models():
    device = get_device()
    print(f"Using device: {device}")
    
    # 加载 D=2 模型
    model2 = VQVAE(latent_dim=2, codebook_K=256).to(device)
    model2.load_state_dict(torch.load(model2_path, map_location=device, weights_only=True))
    model2.eval()
    
    # 加载 D=32 模型
    model32 = VQVAE(latent_dim=32, codebook_K=256).to(device)
    model32.load_state_dict(torch.load(model32_path, map_location=device, weights_only=True))
    model32.eval()
    
    return model2, model32, device

def get_mnist_test_loader(batch_size=10000):
    return get_test_loader(batch_size=batch_size)

def generate_reconstruction_comparison(model2, model32, device):
    print("Generating reconstruction comparison plot...")
    test_loader = get_mnist_test_loader(batch_size=1000)
    images, labels = next(iter(test_loader))
    
    # 找到 0-9 每个数字的第一个样本
    selected_indices = []
    for digit in range(10):
        idx = (labels == digit).nonzero(as_tuple=True)[0][0].item()
        selected_indices.append(idx)
        
    selected_images = images[selected_indices].to(device)
    selected_labels = labels[selected_indices]
    
    with torch.no_grad():
        # D=2 重建
        _, _, _, recon2, _ = model2(selected_images)
        # D=32 重建
        _, _, _, recon32, _ = model32(selected_images)
        
    # 反归一化到 [0, 1]
    orig_imgs = ((selected_images + 1) / 2).cpu().clamp(0, 1).numpy()
    recon2_imgs = ((recon2 + 1) / 2).cpu().clamp(0, 1).numpy()
    recon32_imgs = ((recon32 + 1) / 2).cpu().clamp(0, 1).numpy()
    
    # 计算 MSE 和 PSNR
    def compute_psnr_and_mse(orig, recon):
        mse = np.mean((orig - recon) ** 2, axis=(1, 2, 3))
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
        return mse, psnr
    
    mse2, psnr2 = compute_psnr_and_mse(orig_imgs, recon2_imgs)
    mse32, psnr32 = compute_psnr_and_mse(orig_imgs, recon32_imgs)
    
    # 绘图
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(10, 3, figsize=(10, 20), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    
    # 列标题
    cols = ['Original Image', 'D = 2 Reconstruction', 'D = 32 Reconstruction']
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=14, fontweight='bold', pad=10)
        
    for i in range(10):
        # 原始图片
        ax_orig = axes[i, 0]
        ax_orig.imshow(orig_imgs[i].squeeze(), cmap='gray')
        ax_orig.axis('off')
        ax_orig.text(-5, 14, f"Digit {i}", va='center', ha='right', fontsize=12, fontweight='bold')
        
        # D = 2 重建
        ax_r2 = axes[i, 1]
        ax_r2.imshow(recon2_imgs[i].squeeze(), cmap='gray')
        ax_r2.axis('off')
        ax_r2.text(32, 10, f"MSE: {mse2[i]:.4f}\nPSNR: {psnr2[i]:.2f}dB", va='center', ha='left', fontsize=10, color='#d9534f' if mse2[i] > 0.05 else '#337ab7')
        
        # D = 32 重建
        ax_r32 = axes[i, 2]
        ax_r32.imshow(recon32_imgs[i].squeeze(), cmap='gray')
        ax_r32.axis('off')
        ax_r32.text(32, 10, f"MSE: {mse32[i]:.4f}\nPSNR: {psnr32[i]:.2f}dB", va='center', ha='left', fontsize=10, color='#5cb85c')
        
    # 计算全测试集上的平均 MSE 和 PSNR
    print("Computing full-set metrics...")
    full_loader = get_mnist_test_loader(batch_size=2000)
    all_orig, all_recon2, all_recon32 = [], [], []
    with torch.no_grad():
        for imgs, _ in full_loader:
            imgs_dev = imgs.to(device)
            _, _, _, r2, _ = model2(imgs_dev)
            _, _, _, r32, _ = model32(imgs_dev)
            all_orig.append(((imgs_dev + 1) / 2).cpu().clamp(0, 1))
            all_recon2.append(((r2 + 1) / 2).cpu().clamp(0, 1))
            all_recon32.append(((r32 + 1) / 2).cpu().clamp(0, 1))
            
    all_orig = torch.cat(all_orig, dim=0).numpy()
    all_recon2 = torch.cat(all_recon2, dim=0).numpy()
    all_recon32 = torch.cat(all_recon32, dim=0).numpy()
    
    full_mse2 = np.mean((all_orig - all_recon2) ** 2)
    full_psnr2 = 20 * np.log10(1.0 / np.sqrt(full_mse2))
    full_mse32 = np.mean((all_orig - all_recon32) ** 2)
    full_psnr32 = 20 * np.log10(1.0 / np.sqrt(full_mse32))
    
    plt.suptitle(
        f"VQ-VAE Reconstruction Quality Comparison (K = 256)\n"
        f"Average D=2: MSE = {full_mse2:.4f}, PSNR = {full_psnr2:.2f}dB   |   "
        f"Average D=32: MSE = {full_mse32:.4f}, PSNR = {full_psnr32:.2f}dB",
        fontsize=16, fontweight='bold', y=0.96
    )
    
    plt.subplots_adjust(top=0.92, bottom=0.02, left=0.1, right=0.85, hspace=0.3, wspace=0.3)
    save_path = OUTPUT_DIR / "reconstruction_comparison.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Reconstruction comparison plot saved to {save_path}")

def generate_latent_space_pca(model2, model32, device):
    print("Generating latent space PCA plot...")
    # 用较大的 batch 收集 z_e 隐变量和 label
    test_loader = get_mnist_test_loader(batch_size=5000)
    images, labels = next(iter(test_loader))
    images = images.to(device)
    
    with torch.no_grad():
        # D=2
        z_e2 = model2.encoder(images).cpu().numpy()
        _, _, indices2 = model2.codebook(model2.encoder(images))
        indices2 = indices2.cpu().numpy()
        # D=32
        z_e32 = model32.encoder(images).cpu().numpy()
        _, _, indices32 = model32.codebook(model32.encoder(images))
        indices32 = indices32.cpu().numpy()
        
    labels = labels.numpy()
    
    # 提取 Codebook weights
    codebook2_weights = model2.codebook.codebook.weight.data.cpu().numpy()  # (256, 2)
    codebook32_weights = model32.codebook.codebook.weight.data.cpu().numpy()  # (256, 32)
    
    # 统计 Codebook 激活率 (Codebook Collapse 程度)
    unique_indices2 = np.unique(indices2)
    unique_indices32 = np.unique(indices32)
    
    active_codes2 = len(unique_indices2)
    active_codes32 = len(unique_indices32)
    
    active_mask2 = np.zeros(256, dtype=bool)
    active_mask2[unique_indices2] = True
    
    active_mask32 = np.zeros(256, dtype=bool)
    active_mask32[unique_indices32] = True
    
    # 打印激活状态
    print(f"D=2 Active codes: {active_codes2}/256 ({(active_codes2/256)*100:.1f}%)")
    print(f"D=32 Active codes: {active_codes32}/256 ({(active_codes32/256)*100:.1f}%)")
    
    # D=32 PCA 降维
    pca = PCA(n_components=2)
    z_e32_2d = pca.fit_transform(z_e32)
    codebook32_2d = pca.transform(codebook32_weights)
    explained_var = pca.explained_variance_ratio_.sum()
    
    # 绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    
    # 配色方案
    scatter_cmap = 'tab10'
    
    # ---- 1. D = 2 Latent Space (无需降维) ----
    scatter1 = ax1.scatter(
        z_e2[:, 0], z_e2[:, 1],
        c=labels, cmap=scatter_cmap, s=3, alpha=0.4, label='z_e (Encoder output)'
    )
    # 绘制未激活的 codebook 向量（小灰色圈）
    ax1.scatter(
        codebook2_weights[~active_mask2, 0], codebook2_weights[~active_mask2, 1],
        c='#7f8c8d', marker='o', s=20, edgecolors='black', linewidths=0.5, alpha=0.6, label='Collapsed Codebook Vector'
    )
    # 绘制激活的 codebook 向量（红黄大星号）
    ax1.scatter(
        codebook2_weights[active_mask2, 0], codebook2_weights[active_mask2, 1],
        c='#e74c3c', marker='*', s=120, edgecolors='black', linewidths=0.5, zorder=10, label='Active Codebook Vector'
    )
    
    ax1.set_title(f"D = 2 Latent Space (Original Space)\nActive Codes: {active_codes2}/256 ({(active_codes2/256)*100:.1f}%)", fontsize=15, fontweight='bold', pad=12)
    ax1.set_xlabel("Latent Dim 1", fontsize=12)
    ax1.set_ylabel("Latent Dim 2", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # ---- 2. D = 32 Latent Space (PCA 2D Projection) ----
    scatter2 = ax2.scatter(
        z_e32_2d[:, 0], z_e32_2d[:, 1],
        c=labels, cmap=scatter_cmap, s=3, alpha=0.4, label='z_e (Encoder output)'
    )
    # 绘制未激活的 codebook 向量（小灰色圈）
    ax2.scatter(
        codebook32_2d[~active_mask32, 0], codebook32_2d[~active_mask32, 1],
        c='#7f8c8d', marker='o', s=20, edgecolors='black', linewidths=0.5, alpha=0.6, label='Collapsed Codebook Vector'
    )
    # 绘制激活的 codebook 向量（红黄大星号）
    ax2.scatter(
        codebook32_2d[active_mask32, 0], codebook32_2d[active_mask32, 1],
        c='#e74c3c', marker='*', s=120, edgecolors='black', linewidths=0.5, zorder=10, label='Active Codebook Vector'
    )
    
    ax2.set_title(f"D = 32 Latent Space (PCA 2D Projection)\nActive Codes: {active_codes32}/256 ({(active_codes32/256)*100:.1f}%), Explained Var: {explained_var:.1%}", fontsize=15, fontweight='bold', pad=12)
    ax2.set_xlabel("PC 1", fontsize=12)
    ax2.set_ylabel("PC 2", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    # 添加整体 colorbar
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(scatter2, cax=cbar_ax, ticks=range(10))
    cbar.set_label('MNIST Digit Class', fontsize=12, fontweight='bold', labelpad=10)
    cbar.ax.tick_params(labelsize=10)
    scatter2.set_clim(-0.5, 9.5)
    
    # 添加 Legend
    handles, fig_labels = ax1.get_legend_handles_labels()
    # 我们只显示 z_e, Active, Collapsed 三个 legend
    fig.legend(handles, fig_labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=3, fontsize=12, frameon=True)
    
    plt.suptitle("VQ-VAE Latent Space Visualization & Codebook Distribution (K = 256)\nD = 2 (No collapse) vs D = 32 (Severe Codebook Collapse)", fontsize=18, fontweight='bold', y=0.98)
    
    save_path = OUTPUT_DIR / "latent_space_pca.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Latent space PCA plot saved to {save_path}")

if __name__ == "__main__":
    model2, model32, device = load_models()
    generate_reconstruction_comparison(model2, model32, device)
    generate_latent_space_pca(model2, model32, device)
    print("All tasks finished successfully!")
