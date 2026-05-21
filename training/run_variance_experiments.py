"""
VQ-VAE Codebook Initialization Variance Phase Transition Experiment Runner
-------------------------------------------------------------------------
This script investigates how the codebook collapse phase transition line D* ≈ log2(K)
depends on the codebook initialization variance.

Author: Antigravity AI Assistant
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Set matplotlib backend to non-interactive
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path for imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "models"))

from models.vqvae import VQVAE
from concurrent.futures import ProcessPoolExecutor, as_completed


class GPUDataLoader:
    """
    A fast DataLoader that preloads the entire dataset onto the GPU/MPS device.
    Eliminates CPU-to-GPU copy and dataloader worker overhead during training epochs.
    """
    def __init__(self, dataset, batch_size=64, shuffle=True, device=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device if device is not None else torch.device("cpu")
        
        print(f"[GPU Preloader]: Preloading dataset of size {len(dataset)} onto {self.device}...")
        start_time = time.time()
        
        # Load everything using a temporary standard DataLoader
        temp_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)
        all_images, all_labels = next(iter(temp_loader))
        
        self.images = all_images.to(self.device)
        self.labels = all_labels.to(self.device)
        
        self.num_samples = len(dataset)
        self.num_batches = (self.num_samples + batch_size - 1) // batch_size
        print(f"[GPU Preloader]: Preloaded successfully in {time.time() - start_time:.2f} seconds.")

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(self.num_samples, device=self.device)
        else:
            indices = torch.arange(self.num_samples, device=self.device)
            
        for i in range(0, self.num_samples, self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            yield self.images[batch_indices], self.labels[batch_indices]

    def __len__(self):
        return self.num_batches


def get_device():
    """
    Detects the best available hardware accelerator.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


@torch.no_grad()
def evaluate_metrics(model, data_loader, device, K):
    """
    Performs a deterministic evaluation pass over the entire dataset
    to calculate precise Reconstruction MSE, Codebook Utilization, and Perplexity.
    """
    model.eval()
    all_indices = []
    total_recon_loss = 0.0
    
    for images, _ in data_loader:
        images = images.to(device)
        z_e, z_q_raw, z_q_st, x_recon, indices = model(images)
        
        all_indices.append(indices.cpu())
        total_recon_loss += F.mse_loss(x_recon, images, reduction='sum').item()
        
    all_indices = torch.cat(all_indices, dim=0)
    total_samples = len(data_loader.dataset)
    
    # Normalize by total pixels (1 channel * 28 * 28 pixels)
    avg_recon_loss = total_recon_loss / (total_samples * 1 * 28 * 28)
    
    # Perplexity calculation: P = exp(-sum p_i log p_i)
    counts = torch.bincount(all_indices, minlength=K).float()
    probs = counts / (counts.sum() + 1e-12)
    entropy = -torch.sum(probs * torch.log(probs + 1e-12))
    perplexity = torch.exp(entropy).item()
    
    # Utilization calculation
    unique_used = (counts > 0).sum().item()
    utilization = unique_used / K
    
    return avg_recon_loss, utilization, perplexity


def compile_log_from_results(results_dir, log_path):
    """
    Compiles all individual result JSON files from results_dir into a single unified log dict
    and saves it to log_path.
    """
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


def train_single_model_variance(scale_factor, D, args, device_name, exp_dir):
    """
    Worker function to train a single VQ-VAE model.
    """
    run_id = f"scale{scale_factor}_D{D}"
    results_dir = exp_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    local_result_path = results_dir / f"{run_id}.json"
    
    # Since there's no breakpoint restore required, but checking local file existence is
    # standard practice to allow partial re-runs if interrupted, we'll keep a simple check.
    if local_result_path.exists():
        print(f"--> Skip Completed Run: Scale={scale_factor}, D={D}")
        return

    # Set device
    device = torch.device(device_name)
    
    # Enable hardware-level TensorFloat-32 (TF32) and cuDNN benchmark for Ampere GPUs (RTX 3080 Ti)
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    # Dry run params
    epochs = 1 if args.dry_run else args.epochs
    K = 256
    
    print(f"\n--> [Worker] Starting VQ-VAE Training: Scale={scale_factor}, D={D} for {epochs} epochs on {device}...")
    
    # Load dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    if args.dry_run:
        full_dataset = datasets.MNIST(root=ROOT / 'data', train=True, download=False, transform=transform)
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(len(full_dataset), size=1000, replace=False)
        train_dataset = Subset(full_dataset, indices)
    else:
        train_dataset = datasets.MNIST(root=ROOT / 'data', train=True, download=False, transform=transform)
        
    train_loader = GPUDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, device=device)
    
    # Reseed torch and numpy for weight initialization reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    model = VQVAE(latent_dim=D, codebook_K=K).to(device)
    
    # Scale codebook weight standard deviation
    if scale_factor != 1.0:
        with torch.no_grad():
            model.codebook.codebook.weight.mul_(scale_factor)
            
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    epoch_logs = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        for images, _ in train_loader:
            optimizer.zero_grad()
            z_e, z_q_raw, z_q_st, x_recon, indices = model(images)
            
            recon_loss = F.mse_loss(x_recon, images)
            codebook_loss = F.mse_loss(z_q_raw, z_e.detach())
            commitment_loss = F.mse_loss(z_e, z_q_raw.detach())
            
            loss = codebook_loss + 0.2 * commitment_loss + recon_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        eval_recon, eval_util, eval_perp = evaluate_metrics(model, train_loader, device, K)
        epoch_logs.append({
            "epoch": epoch,
            "train_loss": total_loss / len(train_loader),
            "recon_loss": eval_recon,
            "utilization": eval_util,
            "perplexity": eval_perp
        })
        
        print(f"    [Scale={scale_factor}, D={D}] Epoch {epoch:2d}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Recon MSE: {eval_recon:.4f} | Util: {eval_util*100:5.1f}% | Perp: {eval_perp:.2f}")
        
    # Save model checkpoint
    model_dir = exp_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    local_model_path = model_dir / f"vqvae_{run_id}.pth"
    torch.save(model.state_dict(), local_model_path)
    
    # Save metrics dict to result JSON
    result_data = {
        "scale_factor": scale_factor,
        "D": D,
        "epoch_metrics": epoch_logs,
        "final_metrics": epoch_logs[-1]
    }
    with open(local_result_path, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"    ✓ [Scale={scale_factor}, D={D}] Completed and metrics saved.")


def generate_plots(log_data, plot_dir):
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    scale_factors = [1.0, 0.2, 0.1, 0.05]
    D_list = [4, 6, 8, 10, 12, 16]
    K = 256
    
    plt.figure(figsize=(9, 6))
    
    # High-quality premium aesthetic colors and markers
    colors = {1.0: '#E63946', 0.2: '#457B9D', 0.1: '#1D3557', 0.05: '#A8DADC'}
    labels = {
        1.0: r"$\sigma_0$ (Baseline, 1.0)",
        0.2: r"$\sigma_0 / 5$ (0.2)",
        0.1: r"$\sigma_0 / 10$ (0.1)",
        0.05: r"$\sigma_0 / 20$ (0.05)"
    }
    markers = {1.0: 'o', 0.2: 's', 0.1: '^', 0.05: 'd'}
    
    for scale in scale_factors:
        perps = []
        available_Ds = []
        for D in D_list:
            run_id = f"scale{scale}_D{D}"
            if run_id in log_data["runs"]:
                final = log_data["runs"][run_id]["final_metrics"]
                perps.append(final["perplexity"] / K)
                available_Ds.append(D)
        
        if available_Ds:
            plt.plot(available_Ds, perps, marker=markers[scale], color=colors[scale],
                     linewidth=2.5, markersize=8, label=labels[scale])
            
    # Reference transition point: log2(K) = log2(256) = 8
    plt.axvline(x=8, color='#8D99AE', linestyle='--', linewidth=1.5, alpha=0.8,
                label=r"Theoretical Phase Transition $D^* = \log_2 K = 8$")
                
    # Style formatting
    plt.title("Impact of Codebook Initialization Variance on VQ-VAE Phase Transition", 
              fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Latent Dimension (D)", fontsize=12, labelpad=8)
    plt.ylabel("Normalized Perplexity (Perplexity / K)", fontsize=12, labelpad=8)
    plt.ylim(-0.05, 1.05)
    plt.xticks(D_list)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc="lower left", fontsize=10.5, frameon=True, facecolor='white', edgecolor='#E5E5E5')
    
    plt.tight_layout()
    plot_path = plot_dir / "variance_impact_perplexity.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"  ✓ Phase transition plot saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="VQ-VAE Codebook Initialization Variance Experiments")
    parser.add_argument("--epochs", type=int, default=45, help="Number of epochs for training (default 45)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training (default 64)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for replication (default 42)")
    parser.add_argument("--workers", type=int, default=2, help="Number of parallel experiment workers (default 2)")
    parser.add_argument("--dry-run", action="store_true", help="Perform a quick dry run with minimal epochs/grids")
    args = parser.parse_args()
    
    device = get_device()
    device_name = str(device)
    print(f"\n[Hardware Accelerator]: Detected device = {device_name}")
    
    # Enable hardware-level TensorFloat-32 (TF32) and cuDNN benchmark for Ampere GPUs (RTX 3080 Ti)
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Pre-download dataset to avoid multi-worker race conditions
    print("[Dataset Setup]: Ensuring MNIST dataset is fully downloaded locally...")
    datasets.MNIST(root=ROOT / 'data', train=True, download=True, transform=transform)
    
    exp_dir = ROOT / "outputs" / "collapse_experiment" / "exp_variance"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Scaling factors and scan range for D
    scale_factors = [1.0, 0.2, 0.1, 0.05]
    D_list = [4, 6, 8, 10, 12, 16]
    
    if args.dry_run:
        scale_factors = [1.0, 0.2]
        D_list = [4, 8]
        print("  [DRY RUN] Running with minimal scaling grid.")
        
    tasks = []
    for scale in scale_factors:
        for D in D_list:
            tasks.append((scale, D))
            
    print(f"  Total planned runs: {len(tasks)}. Parallel workers: {args.workers}")
    
    # Run training tasks in parallel
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                train_single_model_variance, scale, D, args, device_name, exp_dir
            ): (scale, D) for scale, D in tasks
        }
        
        for future in as_completed(futures):
            scale, D = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"  [ERROR] Worker failed for Scale={scale}, D={D}: {e}")
                
    # Compile log and generate plot
    local_log_path = exp_dir / "log.json"
    log_data = compile_log_from_results(exp_dir / "results", local_log_path)
    
    print("\nExperiment Training Completed!")
    
    plot_dir = exp_dir / "plots"
    generate_plots(log_data, plot_dir)
    
    print("\n" + "="*80)
    print("  EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"  Local outputs directory: {exp_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
