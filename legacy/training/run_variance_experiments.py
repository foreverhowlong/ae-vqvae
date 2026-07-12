"""
VQ-VAE Codebook Initialization Variance Phase Transition Experiment Runner
-------------------------------------------------------------------------
This script investigates how the codebook collapse phase transition line D* ≈ log2(K)
depends on the codebook initialization variance.

Author: Antigravity AI Assistant
"""

import json
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

# Set matplotlib backend to non-interactive
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import ROOT, get_device, enable_tf32
from legacy.common.data import GPUDataLoader, get_mnist, get_subsampled_mnist
from legacy.common.experiment import compile_log_from_results, train_vqvae
from common.tracking import wandb_run
from legacy.models.vqvae import VQVAE


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
    enable_tf32(device)

    # Dry run params
    epochs = 1 if args.dry_run else args.epochs
    K = 256

    print(f"\n--> [Worker] Starting VQ-VAE Training: Scale={scale_factor}, D={D} for {epochs} epochs on {device}...")

    # Load dataset
    if args.dry_run:
        train_dataset = get_subsampled_mnist(size=1000, seed=args.seed)
    else:
        train_dataset = get_mnist(train=True, download=False)

    train_loader = GPUDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, device=device)

    # Reseed torch and numpy for weight initialization reproducibility
    torch.manual_seed(args.seed)

    model = VQVAE(latent_dim=D, codebook_K=K).to(device)

    # Scale codebook weight standard deviation
    if scale_factor != 1.0:
        with torch.no_grad():
            model.codebook.codebook.weight.mul_(scale_factor)

    with wandb_run(run_id, group="variance", tags=["mnist", "vqvae", "variance"], config={"scale_factor": scale_factor, "latent_dim": D, "codebook_size": K, "epochs": epochs, "batch_size": args.batch_size, "seed": args.seed}):
        epoch_logs = train_vqvae(
            model, train_loader, device, K, epochs,
            log_label=f"Scale={scale_factor}, D={D}",
        )

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
    enable_tf32(device)

    # Pre-download dataset to avoid multi-worker race conditions
    print("[Dataset Setup]: Ensuring MNIST dataset is fully downloaded locally...")
    get_mnist(train=True, download=True)

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
