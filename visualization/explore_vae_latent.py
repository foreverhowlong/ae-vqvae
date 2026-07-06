"""交互式 VAE 潜空间探索器

背景是测试集的潜空间散点图（按数字类别着色），
有一个可拖拽的圆点（knob），拖动时实时解码生成对应的图片。
"""

import torch
import matplotlib.pyplot as plt

from common import ROOT, get_device
from common.data import get_test_loader
from models.vae import VAE

# ── 配置 ──────────────────────────────────────────
model_path = ROOT / "outputs/vae2.pth"


def main():
    device = get_device()

    # ── 加载模型 ──────────────────────────────────
    model = VAE().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # ── 准备测试集潜空间数据 ──────────────────────
    test_loader = get_test_loader(batch_size=256)

    all_latents = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            mu, _ = model.encoder(images)
            all_latents.append(mu.cpu())
            all_labels.append(labels)

    all_latents = torch.cat(all_latents, dim=0).numpy()  # (N, 2)
    all_labels = torch.cat(all_labels, dim=0).numpy()     # (N,)

    # 计算潜空间的范围，留一点边距
    x_min, x_max = all_latents[:, 0].min() - 0.5, all_latents[:, 0].max() + 0.5
    y_min, y_max = all_latents[:, 1].min() - 0.5, all_latents[:, 1].max() + 0.5

    # ── 解码函数 ──────────────────────────────────
    def decode(latent_x, latent_y):
        """给定潜空间坐标，用 decoder 生成图片"""
        with torch.no_grad():
            z = torch.tensor([[latent_x, latent_y]], dtype=torch.float32, device=device)
            img = model.decoder(z)              # (1, 1, 28, 28)
            img = (img + 1) / 2                 # [-1,1] → [0,1]
            img = img.cpu().clamp(0, 1).squeeze()  # (28, 28)
        return img.numpy()

    # ── 构建图面 ──────────────────────────────────
    fig = plt.figure("VAE 潜空间探索器", figsize=(12, 6))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.92)

    # 左侧：散点图
    ax_scatter = fig.add_axes([0.05, 0.08, 0.55, 0.82])
    ax_scatter.set_xlim(x_min, x_max)
    ax_scatter.set_ylim(y_min, y_max)
    ax_scatter.set_xlabel('Latent dim 1')
    ax_scatter.set_ylabel('Latent dim 2')
    ax_scatter.set_title('拖动红点探索潜空间')

    scatter = ax_scatter.scatter(all_latents[:, 0], all_latents[:, 1],
                                 c=all_labels, cmap='tab10', s=2, alpha=0.5)
    cbar = fig.colorbar(scatter, ax=ax_scatter, ticks=range(10), pad=0.01)
    cbar.set_label('Digit class')
    scatter.set_clim(-0.5, 9.5)

    # 初始 knob 位置（潜空间中心）
    init_x = (x_min + x_max) / 2
    init_y = (y_min + y_max) / 2
    (knob,) = ax_scatter.plot(init_x, init_y, 'o', color='red',
                              markersize=12, markeredgecolor='white',
                              markeredgewidth=2, zorder=10)

    # 右侧：生成图片
    ax_img = fig.add_axes([0.68, 0.15, 0.28, 0.7])
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_title('生成的图片')

    init_img = decode(init_x, init_y)
    img_display = ax_img.imshow(init_img, cmap='gray', vmin=0, vmax=1)

    # 坐标文字
    coord_text = ax_scatter.text(0.02, 0.98, '', transform=ax_scatter.transAxes,
                                 fontsize=10, verticalalignment='top',
                                 bbox=dict(boxstyle='round,pad=0.3',
                                           facecolor='white', alpha=0.8))

    # ── 拖拽逻辑 ──────────────────────────────────
    dragging = False

    def on_press(event):
        nonlocal dragging
        if event.inaxes != ax_scatter:
            return
        # 检查是否点击在 knob 附近
        if event.xdata is None:
            return
        dist = ((event.xdata - knob.get_xdata()[0]) ** 2 +
                (event.ydata - knob.get_ydata()[0]) ** 2) ** 0.5
        # 用数据坐标的距离，阈值根据范围自适应
        threshold = max(x_max - x_min, y_max - y_min) * 0.04
        if dist < threshold:
            dragging = True

    def on_release(event):
        nonlocal dragging
        dragging = False

    def on_motion(event):
        nonlocal dragging
        if not dragging or event.inaxes != ax_scatter:
            return
        if event.xdata is None:
            return

        # 更新 knob 位置
        lx = float(event.xdata)
        ly = float(event.ydata)
        knob.set_xdata([lx])
        knob.set_ydata([ly])

        # 解码并更新图片
        img = decode(lx, ly)
        img_display.set_data(img)

        # 更新坐标文字
        coord_text.set_text(f'z = ({lx:.2f}, {ly:.2f})')

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)

    # 初始坐标文字
    coord_text.set_text(f'z = ({init_x:.2f}, {init_y:.2f})')

    plt.show()


if __name__ == "__main__":
    main()
