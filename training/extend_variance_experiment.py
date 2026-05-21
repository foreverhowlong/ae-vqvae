"""
VQ-VAE Codebook Initialization Variance - Experiment Extension Runner
---------------------------------------------------------------------
Resumes training for specific small variance configurations (scale = 0.2, 0.1, 0.05)
at high dimensions (D = 12, 16) from epoch 45 up to epoch 120.

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


def train_single_model_extend(scale_factor, D, target_epochs, args, device_name, exp_dir):
    """
    Worker function to load checkpoint and extend VQ-VAE model training.
    """
    run_id = f"scale{scale_factor}_D{D}"
    results_dir = exp_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    local_result_path = results_dir / f"{run_id}.json"
    
    model_dir = exp_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    local_model_path = model_dir / f"vqvae_{run_id}.pth"
    
    # Check device
    device = torch.device(device_name)
    K = 256
    
    # Enable hardware-level TensorFloat-32 (TF32) and cuDNN benchmark for RTX 3080 Ti
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # Setup baseline for dry run or if files are missing
    if args.dry_run:
        # Create a dummy model and log to simulate existence
        epoch_logs = []
        start_epoch = 1
        model = VQVAE(latent_dim=D, codebook_K=K).to(device)
        if scale_factor != 1.0:
            with torch.no_grad():
                model.codebook.codebook.weight.mul_(scale_factor)
        torch.save(model.state_dict(), local_model_path)
    else:
        if not local_result_path.exists() or not local_model_path.exists():
            print(f"[Worker ERROR] Baseline model or result not found for Scale={scale_factor}, D={D} at {local_model_path}!")
            return
            
        with open(local_result_path, "r") as f:
            result_data = json.load(f)
            
        epoch_logs = result_data.get("epoch_metrics", [])
        start_epoch = len(epoch_logs) + 1
        
        # Load model state dict
        model = VQVAE(latent_dim=D, codebook_K=K).to(device)
        model.load_state_dict(torch.load(local_model_path, map_location=device))

    if start_epoch > target_epochs:
        print(f"--> Scale={scale_factor}, D={D} already trained up to {len(epoch_logs)} epochs. Skipping.")
        return

    print(f"\n--> [Worker] Resuming VQ-VAE Training: Scale={scale_factor}, D={D} from epoch {start_epoch} to {target_epochs} on {device}...")
    
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
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(start_epoch, target_epochs + 1):
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
        
        print(f"    [Scale={scale_factor}, D={D}] Epoch {epoch:3d}/{target_epochs} | Loss: {total_loss/len(train_loader):.4f} | Recon MSE: {eval_recon:.4f} | Util: {eval_util*100:5.1f}% | Perp: {eval_perp:.2f}")
        
    # Save updated model checkpoint
    torch.save(model.state_dict(), local_model_path)
    
    # Save updated result JSON
    updated_result_data = {
        "scale_factor": scale_factor,
        "D": D,
        "epoch_metrics": epoch_logs,
        "final_metrics": epoch_logs[-1]
    }
    with open(local_result_path, "w") as f:
        json.dump(updated_result_data, f, indent=2)
    print(f"    ✓ [Scale={scale_factor}, D={D}] Extension completed up to {target_epochs} epochs.")


def generate_plots(log_data, plot_dir):
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    scale_factors = [1.0, 0.2, 0.1, 0.05]
    D_list = [4, 6, 8, 10, 12, 16]
    K = 256
    
    plt.figure(figsize=(9.5, 6.5))
    
    colors = {1.0: '#E63946', 0.2: '#457B9D', 0.1: '#1D3557', 0.05: '#A8DADC'}
    
    # In the legend, we clearly state the extension
    labels = {
        1.0: r"$\sigma_0$ (Baseline, 45 epochs)",
        0.2: r"$\sigma_0 / 5$ (D=12,16 extended to 120 epochs)",
        0.1: r"$\sigma_0 / 10$ (D=12,16 extended to 120 epochs)",
        0.05: r"$\sigma_0 / 20$ (D=12,16 extended to 120 epochs)"
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
    plt.title("Impact of Initialization Variance on VQ-VAE Phase Transition\n(Extended D=12, 16 with Small Variances to 120 Epochs)", 
              fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Latent Dimension (D)", fontsize=12, labelpad=8)
    plt.ylabel("Normalized Perplexity (Perplexity / K)", fontsize=12, labelpad=8)
    plt.ylim(-0.05, 1.05)
    plt.xticks(D_list)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc="lower left", fontsize=10, frameon=True, facecolor='white', edgecolor='#E5E5E5')
    
    plt.tight_layout()
    plot_path = plot_dir / "variance_impact_extended_perplexity.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"  ✓ Extended phase transition plot saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="VQ-VAE Codebook Initialization Variance Extension Experiments")
    parser.add_argument("--target-epochs", type=int, default=120, help="Total target epochs to train up to (default 120)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training (default 64)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for replication (default 42)")
    parser.add_argument("--workers", type=int, default=2, help="Number of parallel experiment workers (default 2)")
    parser.add_argument("--dry-run", action="store_true", help="Perform a quick dry run with minimal epochs")
    args = parser.parse_args()
    
    device = get_device()
    device_name = str(device)
    print(f"\n[Hardware Accelerator]: Detected device = {device_name}")
    
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
    
    # Targeted scaling factors and dimensions for extension
    scale_factors = [0.2, 0.1, 0.05]
    D_list = [12, 16]
    target_epochs = 2 if args.dry_run else args.target_epochs
    
    if args.dry_run:
        scale_factors = [0.2]
        D_list = [12]
        print("  [DRY RUN] Running extension with minimal grid.")
        
    tasks = []
    for scale in scale_factors:
        for D in D_list:
            tasks.append((scale, D))
            
    print(f"  Starting extension for: {tasks}. Parallel workers: {args.workers}")
    
    # Run training tasks in parallel
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                train_single_model_extend, scale, D, target_epochs, args, device_name, exp_dir
            ): (scale, D) for scale, D in tasks
        }
        
        for future in as_completed(futures):
            scale, D = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"  [ERROR] Extension failed for Scale={scale}, D={D}: {e}")
                
    # Re-compile global log and generate updated plots
    local_log_path = exp_dir / "log.json"
    log_data = compile_log_from_results(exp_dir / "results", local_log_path)
    
    print("\nExperiment Extension Completed!")
    
    plot_dir = exp_dir / "plots"
    generate_plots(log_data, plot_dir)
    
    print("\n" + "="*80)
    print("  EXTENSION COMPLETED SUCCESSFULLY!")
    print(f"  Local outputs directory: {exp_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
