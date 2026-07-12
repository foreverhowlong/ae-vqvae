"""
研究 VQ-VAE 的 Codebook Collapse 现象。

实验设计：
  对不同的 latent_dim（潜空间维数）分别训练 VQ-VAE，
  观察随着维数增加，codebook 利用率、有效秩等指标的变化。

输出：
  outputs/collapse/ 目录下存放：
    - 每个 latent_dim 的模型权重
    - 训练日志（loss、utilization 等）
    - 5 张可视化图
"""

import json
import numpy as np
import torch

from common import ROOT, enable_tf32, get_device
from legacy.common.data import get_test_loader, get_train_loader
from legacy.common.experiment import vq_losses
from common.tracking import log as wandb_log, wandb_run
from legacy.models.vqvae import VQVAE

# ── matplotlib 配置 ─────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")  # 非交互后端，用于保存图片
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════
#  0. 超参数 & 实验配置
# ══════════════════════════════════════════════════════════════

CODEBOOK_K = 256          # codebook 大小（固定）
LATENT_DIMS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]   # 待研究的潜空间维数
EPOCHS = 20
BATCH_SIZE = 64
LR = 1e-3
BETA = 0.2                # commitment loss 权重


def main() -> None:
    # 输出目录
    OUTPUT_DIR = ROOT / "outputs" / "collapse"
    MODEL_DIR = OUTPUT_DIR / "models"
    PLOT_DIR = OUTPUT_DIR / "plots"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    enable_tf32(device)
    print(f"Device: {device}")
    print(f"Output dir: {OUTPUT_DIR}\n")


    # ══════════════════════════════════════════════════════════════
    #  1. 数据加载
    # ══════════════════════════════════════════════════════════════

    train_loader = get_train_loader(batch_size=BATCH_SIZE)
    test_loader = get_test_loader(batch_size=BATCH_SIZE)


    # ══════════════════════════════════════════════════════════════
    #  2. 训练函数
    # ══════════════════════════════════════════════════════════════

    def _train_one_model(latent_dim: int) -> dict:
        """训练一个指定 latent_dim 的 VQ-VAE，返回训练日志。"""
        print(f"{'='*60}")
        print(f"  Training VQ-VAE with latent_dim = {latent_dim}")
        print(f"{'='*60}")

        model = VQVAE(latent_dim, CODEBOOK_K).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        # 日志记录
        log = {
            "latent_dim": latent_dim,
            "epoch": [],
            "recon_loss": [],
            "codebook_loss": [],
            "commitment_loss": [],
            "total_loss": [],
            "codebook_util": [],
            "codebook_counts": None,   # 最后一个 epoch 的计数
        }

        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_recon = 0.0
            total_cb = 0.0
            total_commit = 0.0
            total_loss = 0.0
            all_indices = []

            for images, _ in train_loader:
                images = images.to(device)

                optimizer.zero_grad()

                z_e, z_q_raw, z_q_st, x_recon, indices = model(images)

                loss, recon_loss, codebook_loss, commitment_loss = vq_losses(
                    z_e, z_q_raw, x_recon, images, beta=BETA
                )

                loss.backward()
                optimizer.step()

                total_recon += recon_loss.item()
                total_cb += codebook_loss.item()
                total_commit += commitment_loss.item()
                total_loss += loss.item()
                all_indices.append(indices)

            # ── 统计 ──
            n_batches = len(train_loader)
            all_indices = torch.cat(all_indices, dim=0)  # (total_samples,)
            unique_indices, counts = all_indices.unique(return_counts=True)
            utilization = unique_indices.numel() / CODEBOOK_K

            log["epoch"].append(epoch)
            log["recon_loss"].append(total_recon / n_batches)
            log["codebook_loss"].append(total_cb / n_batches)
            log["commitment_loss"].append(total_commit / n_batches)
            log["total_loss"].append(total_loss / n_batches)
            log["codebook_util"].append(utilization)
            wandb_log({
                "epoch": epoch,
                "train/recon_loss": total_recon / n_batches,
                "train/codebook_loss": total_cb / n_batches,
                "train/commitment_loss": total_commit / n_batches,
                "train/total_loss": total_loss / n_batches,
                "codebook/utilization": utilization,
            }, step=epoch)

            print(
                f"  Epoch {epoch:2d}/{EPOCHS}  "
                f"recon={total_recon/n_batches:.4f}  "
                f"cb={total_cb/n_batches:.4f}  "
                f"commit={total_commit/n_batches:.4f}  "
                f"util={utilization*100:5.1f}%  "
                f"({unique_indices.numel()}/{CODEBOOK_K})"
            )

            # 最后一个 epoch 保存计数
            if epoch == EPOCHS:
                count_array = np.zeros(CODEBOOK_K, dtype=np.int64)
                count_array[unique_indices.cpu().numpy()] = counts.cpu().numpy()
                log["codebook_counts"] = count_array.tolist()

        # ── 保存模型 ──
        model_path = MODEL_DIR / f"vqvae_dim{latent_dim}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"  Model saved → {model_path}\n")

        return log


    def train_one_model(latent_dim: int) -> dict:
        config = {"latent_dim": latent_dim, "codebook_size": CODEBOOK_K, "epochs": EPOCHS,
                  "batch_size": BATCH_SIZE, "lr": LR, "beta": BETA}
        with wandb_run(f"collapse-D{latent_dim}", group="collapse-dimensions",
                       tags=["mnist", "vqvae", "collapse"], config=config):
            return _train_one_model(latent_dim)


    # ══════════════════════════════════════════════════════════════
    #  3. 运行所有实验
    # ══════════════════════════════════════════════════════════════

    all_logs = {}
    for dim in LATENT_DIMS:
        log = train_one_model(dim)
        all_logs[str(dim)] = log

    # 保存日志
    log_path = OUTPUT_DIR / "training_logs.json"
    with open(log_path, "w") as f:
        json.dump(all_logs, f, indent=2)
    print(f"Logs saved → {log_path}")


    # ══════════════════════════════════════════════════════════════
    #  4. 可视化
    # ══════════════════════════════════════════════════════════════

    # ── 辅助：计算 codebook 有效秩（基于 SVD） ──────────────────
    def compute_effective_rank(codebook_weight: np.ndarray) -> float:
        """
        计算 codebook 的有效秩（基于奇异值分布的熵）。
        有效秩 = exp(-sum(s_i' * log(s_i')))，
        其中 s_i' = s_i / sum(s_i) 是归一化奇异值。
        有效秩 ∈ [1, min(K, D)]。
        """
        u, s, vh = np.linalg.svd(codebook_weight, full_matrices=False)
        s_norm = s / (s.sum() + 1e-12)
        entropy = -np.sum(s_norm * np.log(s_norm + 1e-12))
        effective_rank = np.exp(entropy)
        return effective_rank


    # ── 收集各 latent_dim 的最终指标 ────────────────────────────
    dims = np.array(LATENT_DIMS)
    final_recon = []
    final_commit = []
    final_util = []
    effective_ranks = []
    codebook_counts_list = []

    for d in LATENT_DIMS:
        log = all_logs[str(d)]
        final_recon.append(log["recon_loss"][-1])
        final_commit.append(log["commitment_loss"][-1])
        final_util.append(log["codebook_util"][-1])

        # 加载模型权重计算有效秩
        model = VQVAE(d, CODEBOOK_K)
        model_path = MODEL_DIR / f"vqvae_dim{d}.pth"
        model.load_state_dict(torch.load(model_path, weights_only=True))
        cb_weight = model.codebook.codebook.weight.data.cpu().numpy()  # (K, D)
        eff_rank = compute_effective_rank(cb_weight)
        effective_ranks.append(eff_rank)

        codebook_counts_list.append(np.array(log["codebook_counts"]))

    final_recon = np.array(final_recon)
    final_commit = np.array(final_commit)
    final_util = np.array(final_util)
    effective_ranks = np.array(effective_ranks)


    # ── 图 1：latent_dim vs reconstruction loss / commitment loss ──
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(dims, final_recon, "bo-", label="Reconstruction Loss")
    ax1.set_xlabel("Latent Dimension")
    ax1.set_ylabel("Reconstruction Loss", color="b")
    ax1.tick_params(axis="y", labelcolor="b")

    ax2 = ax1.twinx()
    ax2.plot(dims, final_commit, "rs-", label="Commitment Loss")
    ax2.set_ylabel("Commitment Loss", color="r")
    ax2.tick_params(axis="y", labelcolor="r")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.set_title("Fig1: Reconstruction & Commitment Loss vs Latent Dim")
    ax1.set_xticks(dims)
    fig1.tight_layout()
    fig1.savefig(PLOT_DIR / "fig1_loss_vs_dim.png", dpi=150)
    plt.close(fig1)
    print("Fig1 saved.")


    # ── 图 2：latent_dim vs codebook utilization ──────────────────
    fig2, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dims, final_util * 100, "go-", linewidth=2, markersize=8)
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5, label="100% (full)")
    ax.set_xlabel("Latent Dimension")
    ax.set_ylabel("Codebook Utilization (%)")
    ax.set_title("Fig2: Codebook Utilization vs Latent Dim")
    ax.set_xticks(dims)
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(PLOT_DIR / "fig2_utilization_vs_dim.png", dpi=150)
    plt.close(fig2)
    print("Fig2 saved.")


    # ── 图 3：codebook 使用频率分布直方图（每个 dim 一个子图） ──
    n_dims = len(LATENT_DIMS)
    n_cols = 3
    n_rows = (n_dims + n_cols - 1) // n_cols
    fig3, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    TOP_K = 30  # 只显示使用频率最高的前 30 个

    for idx, (d, counts) in enumerate(zip(LATENT_DIMS, codebook_counts_list)):
        ax = axes[idx]
        # 按使用次数降序排列
        sorted_counts = np.sort(counts)[::-1]
        top_counts = sorted_counts[:TOP_K]
        x = np.arange(TOP_K)

        bars = ax.bar(x, top_counts, color="steelblue", edgecolor="navy", linewidth=0.5)
        ax.set_title(f"latent_dim = {d}")
        ax.set_xlabel("Codebook Index (sorted by freq)")
        ax.set_ylabel("Usage Count")
        ax.set_xticks(x[::5])
        ax.set_xticklabels(x[::5])

        # 标注利用率
        used = (counts > 0).sum()
        ax.text(
            0.95, 0.95,
            f"Used: {used}/{CODEBOOK_K}\n({used/CODEBOOK_K*100:.1f}%)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    # 隐藏多余的子图
    for idx in range(n_dims, len(axes)):
        axes[idx].axis("off")

    fig3.suptitle("Fig3: Codebook Usage Frequency (Top-{} per latent dim)".format(TOP_K),
                  fontsize=14, y=1.02)
    fig3.tight_layout()
    fig3.savefig(PLOT_DIR / "fig3_codebook_histogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print("Fig3 saved.")


    # ── 图 4：latent_dim vs effective rank ────────────────────────
    fig4, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dims, effective_ranks, "mo-", linewidth=2, markersize=8, label="Effective Rank")
    ax.plot(dims, dims, "k--", alpha=0.4, label="y = x (full rank)")
    ax.set_xlabel("Latent Dimension")
    ax.set_ylabel("Effective Rank of Codebook")
    ax.set_title("Fig4: Effective Rank vs Latent Dim")
    ax.set_xticks(dims)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig4.tight_layout()
    fig4.savefig(PLOT_DIR / "fig4_effective_rank_vs_dim.png", dpi=150)
    plt.close(fig4)
    print("Fig4 saved.")


    # ── 图 5：不同 latent_dim 的重建质量对比 ─────────────────────
    NUM_SAMPLES = 5  # 每个 dim 展示 5 个样本

    # 从测试集取固定样本
    test_iter = iter(test_loader)
    fixed_images, fixed_labels = next(test_iter)
    fixed_images = fixed_images[:NUM_SAMPLES].to(device)

    # 收集每个 dim 的重建结果
    reconstructions = {}  # dim -> (B, 1, 28, 28) numpy
    for d in LATENT_DIMS:
        model = VQVAE(d, CODEBOOK_K).to(device)
        model_path = MODEL_DIR / f"vqvae_dim{d}.pth"
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        with torch.no_grad():
            _, _, _, x_recon, _ = model(fixed_images)
        # 反归一化 [-1,1] → [0,1]
        reconstructions[d] = ((x_recon + 1) / 2).cpu().numpy()

    # 反归一化原图
    fixed_display = ((fixed_images + 1) / 2).cpu().numpy()

    # 排版：每行 = 1 个样本，每列 = 1 个 latent_dim + 原图
    n_cols = len(LATENT_DIMS) + 1  # +1 为原图列
    fig5, axes = plt.subplots(NUM_SAMPLES, n_cols, figsize=(2.5 * n_cols, 2.5 * NUM_SAMPLES))

    for row in range(NUM_SAMPLES):
        # 第一列：原图
        ax = axes[row, 0]
        ax.imshow(fixed_display[row].squeeze(), cmap="gray")
        ax.set_title(f"Original\n({fixed_labels[row].item()})", fontsize=9)
        ax.axis("off")

        # 后续列：各 latent_dim 的重建
        for col, d in enumerate(LATENT_DIMS):
            ax = axes[row, col + 1]
            ax.imshow(reconstructions[d][row].squeeze(), cmap="gray")
            ax.set_title(f"dim={d}", fontsize=9)
            ax.axis("off")

    fig5.suptitle("Fig5: Reconstruction Quality vs Latent Dim", fontsize=14, y=1.01)
    fig5.tight_layout()
    fig5.savefig(PLOT_DIR / "fig5_reconstruction_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig5)
    print("Fig5 saved.")


    # ══════════════════════════════════════════════════════════════
    #  5. 打印汇总
    # ══════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"  {'latent_dim':>10}  {'recon_loss':>12}  {'commit_loss':>12}  "
          f"{'util%':>8}  {'eff_rank':>10}")
    print("  " + "-" * 58)
    for i, d in enumerate(LATENT_DIMS):
        print(f"  {d:>10d}  {final_recon[i]:>12.4f}  {final_commit[i]:>12.4f}  "
              f"{final_util[i]*100:>7.1f}%  {effective_ranks[i]:>10.2f}")
    print("=" * 70)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print(f"  Models: {MODEL_DIR}/")
    print(f"  Plots:  {PLOT_DIR}/")


if __name__ == "__main__":
    main()
