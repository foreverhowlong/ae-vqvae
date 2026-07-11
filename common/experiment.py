"""VQ-VAE 实验公共代码：损失、训练循环、评估指标、结果日志与 Google Drive 同步。"""

import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common.tracking import log as wandb_log


def vq_losses(z_e, z_q_raw, x_recon, x, beta=0.2):
    """标准 VQ-VAE 三项损失，返回 (total, recon, codebook, commitment)。"""
    recon_loss = F.mse_loss(x_recon, x)
    codebook_loss = F.mse_loss(z_q_raw, z_e.detach())
    commitment_loss = F.mse_loss(z_e, z_q_raw.detach())
    total = codebook_loss + beta * commitment_loss + recon_loss
    return total, recon_loss, codebook_loss, commitment_loss


@torch.no_grad()
def evaluate_metrics(model, data_loader, device, K):
    """对整个数据集做一次确定性评估。

    返回 (avg_recon_loss, utilization, perplexity)。
    """
    model.eval()
    all_indices = []
    total_recon_loss = 0.0

    for images, _ in data_loader:
        images = images.to(device)
        z_e, z_q_raw, z_q_st, x_recon, indices = model(images)

        all_indices.append(indices.cpu())
        total_recon_loss += F.mse_loss(x_recon, images, reduction="sum").item()

    all_indices = torch.cat(all_indices, dim=0)
    total_samples = len(data_loader.dataset)

    # 按总像素数归一化（1 通道 × 28 × 28）
    avg_recon_loss = total_recon_loss / (total_samples * 1 * 28 * 28)

    # Perplexity: P = exp(-sum p_i log p_i)
    counts = torch.bincount(all_indices, minlength=K).float()
    probs = counts / (counts.sum() + 1e-12)
    entropy = -torch.sum(probs * torch.log(probs + 1e-12))
    perplexity = torch.exp(entropy).item()

    unique_used = (counts > 0).sum().item()
    utilization = unique_used / K

    return avg_recon_loss, utilization, perplexity


def train_vqvae(model, train_loader, device, K, target_epochs, *,
                start_epoch=1, lr=1e-3, beta=0.2, log_label="", epoch_logs=None):
    """训练 VQ-VAE 若干个 epoch，每个 epoch 结束后做一次全量评估。

    train_loader 的 batch 需已在 device 上（配合 GPUDataLoader 使用零拷贝）。
    epoch_logs 传入已有日志列表可用于断点续训，返回追加后的日志列表。
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epoch_logs = epoch_logs if epoch_logs is not None else []

    for epoch in range(start_epoch, target_epochs + 1):
        model.train()
        total_loss = 0.0

        for images, _ in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            z_e, z_q_raw, z_q_st, x_recon, indices = model(images)

            loss, _, _, _ = vq_losses(z_e, z_q_raw, x_recon, images, beta=beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        eval_recon, eval_util, eval_perp = evaluate_metrics(model, train_loader, device, K)
        epoch_logs.append({
            "epoch": epoch,
            "train_loss": total_loss / len(train_loader),
            "recon_loss": eval_recon,
            "utilization": eval_util,
            "perplexity": eval_perp,
        })
        wandb_log(epoch_logs[-1], step=epoch)

        print(f"    [{log_label}] Epoch {epoch:3d}/{target_epochs} | "
              f"Loss: {total_loss/len(train_loader):.4f} | Recon MSE: {eval_recon:.4f} | "
              f"Util: {eval_util*100:5.1f}% | Perp: {eval_perp:.2f}")

    return epoch_logs


# ─────────────────────────────────────────────────────────────
#  结果文件管理 & Google Drive 备份
# ─────────────────────────────────────────────────────────────

def compile_log_from_results(results_dir, log_path):
    """把 results_dir 下所有单次运行的 JSON 汇总成一个 log 字典并保存到 log_path。"""
    log_data = {"completed_runs": [], "runs": {}}
    if not results_dir.exists():
        return log_data

    for file_path in results_dir.glob("*.json"):
        run_id = file_path.stem
        try:
            with open(file_path, "r") as f:
                run_data = json.load(f)
            log_data["runs"][run_id] = run_data
            log_data["completed_runs"].append(run_id)
        except Exception as e:
            print(f"  [Compile WARNING] Failed to read {file_path.name}: {e}")

    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)

    return log_data


def sync_file_to_drive(local_path, drive_path):
    """把本地文件备份到 Google Drive 路径（drive_path 为 None 时跳过）。"""
    if drive_path:
        try:
            drive_path = Path(drive_path)
            drive_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_path, drive_path)
            print(f"  [Drive Sync] Successfully backed up to: {drive_path}")
        except Exception as e:
            print(f"  [Drive Sync WARNING] Failed to copy to Google Drive: {e}")


def sync_results_from_drive(local_results_dir, drive_results_dir):
    """把 Google Drive 上已有的结果文件同步回本地，支持并行 worker 的断点续跑。"""
    if not drive_results_dir or not Path(drive_results_dir).exists():
        return

    local_results_dir.mkdir(parents=True, exist_ok=True)
    for drive_file in Path(drive_results_dir).glob("*.json"):
        local_file = local_results_dir / drive_file.name
        if not local_file.exists():
            try:
                shutil.copy2(drive_file, local_file)
                print(f"  [Resuming] Synced {drive_file.name} from Google Drive.")
            except Exception as e:
                print(f"  [WARNING] Failed to sync {drive_file.name}: {e}")


def load_log(local_path, drive_path=None):
    """读取训练日志，优先使用 Google Drive 备份以支持恢复。"""
    if drive_path and Path(drive_path).exists():
        try:
            shutil.copy2(drive_path, local_path)
            print(f"  [Resuming] Synchronized log.json from Google Drive to local.")
        except Exception as e:
            print(f"  [WARNING] Failed to sync log.json from Google Drive: {e}")

    if Path(local_path).exists():
        try:
            with open(local_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"  [Error] Failed to read local log.json: {e}. Starting fresh.")

    return {"completed_runs": [], "runs": {}}
