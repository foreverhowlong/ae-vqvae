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
from matplotlib.gridspec import GridSpec
from vqvae import VQVAE

# 模型参数（与训练时一致）
latent_dim = 2
codebook_K = 256
model_path = ROOT / "outputs/vqvae2.pth"


def get_device():
    return torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )


def load_model(pth_path=model_path):
    device = get_device()
    model = VQVAE(latent_dim, codebook_K).to(device)
    model.load_state_dict(torch.load(pth_path, map_location=device, weights_only=True))
    model.eval()
    return model, device


def get_test_data(batch_size=256):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    test_dataset = datasets.MNIST(root=ROOT / 'data', train=False,
                                  download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader


# ---------------------------------------------------------------------------
# 功能 1：z_e 散点图 + codebook 向量散点图 + 点击 codebook 向量解码
# ---------------------------------------------------------------------------
def plot_latent_space_with_codebook(pth_path=model_path):
    """绘制 z_e 散点图和 codebook 向量散点图，支持点击 codebook 向量查看解码结果。"""
    model, device = load_model(pth_path)
    test_loader = get_test_data()

    # 收集所有 z_e 和标签
    all_z_e = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            z_e, _, _, _, _ = model(images)
            all_z_e.append(z_e.cpu())
            all_labels.append(labels)

    all_z_e = torch.cat(all_z_e, dim=0).numpy()    # (N, 2)
    all_labels = torch.cat(all_labels, dim=0).numpy()  # (N,)

    # 获取 codebook 向量
    codebook_weights = model.codebook.codebook.weight.data.cpu().numpy()  # (K, 2)

    # 创建图形：左侧散点图，右侧解码结果
    fig = plt.figure(figsize=(14, 7))
    gs = GridSpec(1, 2, width_ratios=[2, 1], figure=fig)
    ax_scatter = fig.add_subplot(gs[0])
    ax_decode = fig.add_subplot(gs[1])

    # 绘制 z_e 散点图（按数字类别着色）
    scatter_ze = ax_scatter.scatter(
        all_z_e[:, 0], all_z_e[:, 1],
        c=all_labels, cmap='tab10', s=2, alpha=0.5, label='z_e'
    )
    plt.colorbar(scatter_ze, ax=ax_scatter, ticks=range(10), label='Digit class')
    scatter_ze.set_clim(-0.5, 9.5)

    # 绘制 codebook 向量（红色星形标记）
    codebook_scatter = ax_scatter.scatter(
        codebook_weights[:, 0], codebook_weights[:, 1],
        c='red', marker='*', s=80, zorder=5, label='Codebook vectors'
    )

    ax_scatter.set_xlabel('Latent dim 1')
    ax_scatter.set_ylabel('Latent dim 2')
    ax_scatter.set_title('VQ-VAE Latent Space: z_e (colored) & Codebook (red ★)')
    ax_scatter.legend(loc='upper right', fontsize=8)

    # 右侧初始提示
    ax_decode.text(0.5, 0.5, 'Click a codebook\nvector (red ★)\nto decode',
                   ha='center', va='center', fontsize=14,
                   transform=ax_decode.transAxes)
    ax_decode.set_title('Decoded Codebook Vector')
    ax_decode.axis('off')

    # 点击事件处理
    def on_click(event):
        if event.inaxes != ax_scatter:
            return

        click_x, click_y = event.xdata, event.ydata
        if click_x is None or click_y is None:
            return

        # 计算点击位置与所有 codebook 向量的距离
        distances = ((codebook_weights[:, 0] - click_x) ** 2 +
                     (codebook_weights[:, 1] - click_y) ** 2) ** 0.5
        min_idx = distances.argmin()
        min_dist = distances[min_idx]

        # 只有在点击位置足够接近 codebook 向量时才解码
        # 用 codebook 向量之间平均距离的一定比例作为阈值
        threshold = 0.5
        if min_dist > threshold:
            return

        # 解码该 codebook 向量
        with torch.no_grad():
            z_q = model.codebook.codebook.weight[min_idx].unsqueeze(0).to(device)  # (1, 2)
            decoded = model.decoder(z_q)  # (1, 1, 28, 28)
            decoded_img = (decoded + 1) / 2  # 反归一化到 [0, 1]
            decoded_img = decoded_img.cpu().squeeze().clamp(0, 1).numpy()

        # 更新右侧子图
        ax_decode.clear()
        ax_decode.imshow(decoded_img, cmap='gray')
        ax_decode.set_title(f'Codebook #{min_idx}\n'
                            f'({codebook_weights[min_idx, 0]:.2f}, '
                            f'{codebook_weights[min_idx, 1]:.2f})')
        ax_decode.axis('off')

        # 高亮被选中的 codebook 向量
        # 先重置所有 codebook 向量颜色
        codebook_scatter.set_sizes(torch.full((codebook_K,), 80.0).numpy())
        # 放大被选中的
        sizes = codebook_scatter.get_sizes()
        sizes[min_idx] = 250
        codebook_scatter.set_sizes(sizes)

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 功能 2：5 个样本的原图与重建图对比
# ---------------------------------------------------------------------------
def show_reconstruction(pth_path=model_path, num_pairs=5):
    """加载训练好的 VQ-VAE 模型，从测试集取样本，
    并排显示原图和重建图。

    Args:
        pth_path: 模型权重文件路径
        num_pairs: 显示的原图-重建图对数（默认 5）
    """
    model, device = load_model(pth_path)
    test_loader = get_test_data(batch_size=64)

    # 取一个 batch
    images, labels = next(iter(test_loader))
    images = images[:num_pairs].to(device)
    labels = labels[:num_pairs]

    with torch.no_grad():
        _, _, _, x_recon, _ = model(images)

    # 反归一化：[-1, 1] → [0, 1]
    images = ((images + 1) / 2).cpu().clamp(0, 1)
    x_recon = ((x_recon + 1) / 2).cpu().clamp(0, 1)

    fig, axes = plt.subplots(num_pairs, 2, figsize=(4, num_pairs * 2))

    for i in range(num_pairs):
        # 原图
        ax_orig = axes[i, 0]
        ax_orig.imshow(images[i].squeeze(), cmap="gray")
        ax_orig.set_title(f"Original (label={labels[i].item()})")
        ax_orig.axis("off")

        # 重建图
        ax_recon = axes[i, 1]
        ax_recon.imshow(x_recon[i].squeeze(), cmap="gray")
        ax_recon.set_title("Reconstructed")
        ax_recon.axis("off")

    plt.suptitle('VQ-VAE Reconstruction Comparison', fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 功能 1：交互式散点图 + 点击解码
    plot_latent_space_with_codebook()

    # 功能 2：5 个样本的原图与重建图对比
    show_reconstruction()()