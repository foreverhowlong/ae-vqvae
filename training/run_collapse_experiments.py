"""
VQ-VAE Codebook Collapse Phase Transition Experiment Runner
----------------------------------------------------------
This script runs two systematic experiments to study the codebook collapse phase transition
in VQ-VAE models trained on sub-sampled MNIST. It is designed to run seamlessly on local
machines and Google Colab, supporting automated Google Drive backups and resumable training.

Author: Antigravity AI Assistant
"""

import json
import shutil
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

# Set matplotlib backend to non-interactive to prevent issues in headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import ROOT, get_device, enable_tf32
from common.data import GPUDataLoader, get_mnist, get_subsampled_mnist
from common.experiment import (
    compile_log_from_results,
    sync_file_to_drive,
    sync_results_from_drive,
    train_vqvae,
)
from models.vqvae import VQVAE


# =====================================================================
# Parallel Worker Functions (Picklable Module-Level Targets)
# =====================================================================

def train_single_model_exp1(K, D, args, device_name, exp1_dir, drive_dir):
    """
    Worker function to train a single VQ-VAE model for Experiment 1 in parallel.
    """
    run_id = f"K{K}_D{D}"
    results_dir = exp1_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    local_result_path = results_dir / f"{run_id}.json"
    drive_result_path = Path(drive_dir) / "exp1" / "results" / f"{run_id}.json" if drive_dir else None
    
    # Check if result file already exists (locally or on Drive)
    if local_result_path.exists():
        print(f"--> Skip Completed Run: K={K}, D={D}")
        return
    if drive_result_path and drive_result_path.exists():
        # Sync from Drive to local
        try:
            local_result_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(drive_result_path, local_result_path)
            print(f"--> Skip Completed Run (synced from Drive): K={K}, D={D}")
            return
        except Exception as e:
            print(f"  [WARNING] Failed to sync {run_id}.json from Google Drive: {e}")

    # Set device
    device = torch.device(device_name)
    enable_tf32(device)
    
    # Dry run params
    epochs = 1 if args.dry_run else args.epochs1
    
    print(f"\n--> [Worker] Starting VQ-VAE Training: K={K}, D={D} for {epochs} epochs on {device}...")
    
    if args.dry_run:
        train_dataset = get_subsampled_mnist(size=1000, seed=args.seed)
    else:
        train_dataset = get_mnist(train=True, download=False)
        
    train_loader = GPUDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, device=device)
    
    # Reseed torch and numpy for weight initialization reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    model = VQVAE(latent_dim=D, codebook_K=K).to(device)
    epoch_logs = train_vqvae(
        model, train_loader, device, K, epochs,
        log_label=f"K={K}, D={D}",
    )
        
    # Save model checkpoint
    model_dir = exp1_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    local_model_path = model_dir / f"vqvae_{run_id}.pth"
    drive_model_path = Path(drive_dir) / "exp1" / "models" / f"vqvae_{run_id}.pth" if drive_dir else None
    
    torch.save(model.state_dict(), local_model_path)
    sync_file_to_drive(local_model_path, drive_model_path)
    
    # Save metrics dict to result JSON
    result_data = {
        "K": K,
        "D": D,
        "epoch_metrics": epoch_logs,
        "final_metrics": epoch_logs[-1]
    }
    with open(local_result_path, "w") as f:
        json.dump(result_data, f, indent=2)
    sync_file_to_drive(local_result_path, drive_result_path)
    print(f"    ✓ [K={K}, D={D}] Completed and metrics saved.")


def train_single_model_exp2(seed, D, args, device_name, exp2_dir, drive_dir):
    """
    Worker function to train a single VQ-VAE model for Experiment 2 in parallel.
    """
    run_id = f"seed{seed}_D{D}"
    results_dir = exp2_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    local_result_path = results_dir / f"{run_id}.json"
    drive_result_path = Path(drive_dir) / "exp2" / "results" / f"{run_id}.json" if drive_dir else None
    
    # Check if result file already exists (locally or on Drive)
    if local_result_path.exists():
        print(f"--> Skip Completed Run: Seed={seed}, D={D}")
        return
    if drive_result_path and drive_result_path.exists():
        # Sync from Drive to local
        try:
            local_result_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(drive_result_path, local_result_path)
            print(f"--> Skip Completed Run (synced from Drive): Seed={seed}, D={D}")
            return
        except Exception as e:
            print(f"  [WARNING] Failed to sync {run_id}.json from Google Drive: {e}")

    # Set device
    device = torch.device(device_name)
    enable_tf32(device)
    
    # Parameters
    K = 256
    epochs = 1 if args.dry_run else args.epochs2
    
    print(f"\n--> [Worker] Starting VQ-VAE Training: Seed={seed}, D={D} for {epochs} epochs on {device}...")
    
    if args.dry_run:
        train_dataset = get_subsampled_mnist(size=1000, seed=args.seed)
    else:
        train_dataset = get_mnist(train=True, download=False)
        
    train_loader = GPUDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, device=device)
    
    # Reseed torch and numpy for weight initialization reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = VQVAE(latent_dim=D, codebook_K=K).to(device)
    epoch_logs = train_vqvae(
        model, train_loader, device, K, epochs,
        log_label=f"Seed={seed}, D={D}",
    )
        
    # Save model checkpoint
    model_dir = exp2_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    local_model_path = model_dir / f"vqvae_{run_id}.pth"
    drive_model_path = Path(drive_dir) / "exp2" / "models" / f"vqvae_{run_id}.pth" if drive_dir else None
    
    torch.save(model.state_dict(), local_model_path)
    sync_file_to_drive(local_model_path, drive_model_path)
    
    # Save metrics dict to result JSON
    result_data = {
        "seed": seed,
        "D": D,
        "epoch_metrics": epoch_logs,
        "final_metrics": epoch_logs[-1]
    }
    with open(local_result_path, "w") as f:
        json.dump(result_data, f, indent=2)
    sync_file_to_drive(local_result_path, drive_result_path)
    print(f"    ✓ [Seed={seed}, D={D}] Completed and metrics saved.")


# =====================================================================
# 4. Experiment Master Runners
# =====================================================================

def run_experiment_1(args, device_name, test_dataset, output_dir, drive_dir):
    print("\n" + "="*80)
    print("  RUNNING EXPERIMENT 1 (Grid of K x D Phase Transition - PARALLEL)")
    print("="*80)
    
    exp1_dir = output_dir / "exp1"
    local_results_dir = exp1_dir / "results"
    drive_results_dir = Path(drive_dir) / "exp1" / "results" if drive_dir else None
    
    # 1. Sync any existing results from Drive to Local to support resume
    sync_results_from_drive(local_results_dir, drive_results_dir)
    
    # Experiment Grid
    K_list = [64, 256, 1024]
    D_list = [4, 6, 8, 10, 12, 16, 24]
    
    if args.dry_run:
        K_list = [64]
        D_list = [4, 8]
        print("  [DRY RUN] Running with minimal grid sizes.")
        
    # Generate combinations
    tasks = []
    for K in K_list:
        for D in D_list:
            tasks.append((K, D))
            
    print(f"  Total planned runs: {len(tasks)}. Parallel workers: {args.workers}")
    
    # 2. Run training tasks in parallel
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                train_single_model_exp1, K, D, args, device_name, exp1_dir, drive_dir
            ): (K, D) for K, D in tasks
        }
        
        for future in as_completed(futures):
            K, D = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"  [ERROR] Worker failed for K={K}, D={D}: {e}")
                
    # 3. Compile all individual result JSONs into log.json
    local_log_path = exp1_dir / "log.json"
    drive_log_path = Path(drive_dir) / "exp1" / "log.json" if drive_dir else None
    
    log_data = compile_log_from_results(local_results_dir, local_log_path)
    sync_file_to_drive(local_log_path, drive_log_path)
    
    print("\nExperiment 1 Training Completed!")
    
    # 4. Generate Visualizations (runs in parent process)
    plot_dir = exp1_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    model_dir = exp1_dir / "models"
    
    # We load device here to load saved models for test set reconstruction plotting
    device = torch.device(device_name)
    generate_exp1_plots(log_data, model_dir, plot_dir, test_dataset, device, drive_dir)


def run_experiment_2(args, device_name, output_dir, drive_dir):
    print("\n" + "="*80)
    print("  RUNNING EXPERIMENT 2 (Stochasticity Across Random Seeds - PARALLEL)")
    print("="*80)
    
    exp2_dir = output_dir / "exp2"
    local_results_dir = exp2_dir / "results"
    drive_results_dir = Path(drive_dir) / "exp2" / "results" if drive_dir else None
    
    # 1. Sync any existing results from Drive to Local to support resume
    sync_results_from_drive(local_results_dir, drive_results_dir)
    
    # Parameters
    D_list = [8, 9, 10]
    seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]  # 10 random seeds
    
    if args.dry_run:
        D_list = [8]
        seeds = [42, 43]
        print("  [DRY RUN] Running with minimal seeds and dimensions.")
        
    # Generate combinations
    tasks = []
    for D in D_list:
        for seed in seeds:
            tasks.append((seed, D))
            
    print(f"  Total planned runs: {len(tasks)}. Parallel workers: {args.workers}")
    
    # 2. Run training tasks in parallel
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                train_single_model_exp2, seed, D, args, device_name, exp2_dir, drive_dir
            ): (seed, D) for seed, D in tasks
        }
        
        for future in as_completed(futures):
            seed, D = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"  [ERROR] Worker failed for Seed={seed}, D={D}: {e}")
                
    # 3. Compile all individual result JSONs into log.json
    local_log_path = exp2_dir / "log.json"
    drive_log_path = Path(drive_dir) / "exp2" / "log.json" if drive_dir else None
    
    log_data = compile_log_from_results(local_results_dir, local_log_path)
    sync_file_to_drive(local_log_path, drive_log_path)
    
    print("\nExperiment 2 Training Completed!")
    
    # 4. Generate Visualizations (runs in parent process)
    plot_dir = exp2_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    generate_exp2_plots(log_data, plot_dir, drive_dir)


# =====================================================================
# 4. Plotting & Analysis Functions (Matplotlib)
# =====================================================================

def find_crossing(Ds, utils, target):
    """
    Interpolates utilization vs D to find the exact point where utilization crosses a target.
    """
    if len(Ds) < 2:
        return float(Ds[0])
    if utils[0] <= target:
        return float(Ds[0])
    if utils[-1] >= target:
        return float(Ds[-1])
        
    for i in range(len(Ds) - 1):
        u1, u2 = utils[i], utils[i+1]
        d1, d2 = Ds[i], Ds[i+1]
        if (u1 >= target >= u2) or (u1 <= target <= u2):
            # Linear interpolation
            t = (target - u1) / (u2 - u1 + 1e-12)
            return float(d1 + t * (d2 - d1))
            
    return float(Ds[len(Ds)//2])


def generate_exp1_plots(log_data, model_dir, plot_dir, test_dataset, device, drive_dir):
    print("\nGenerating Experiment 1 figures...")
    
    # ── Parse Grid Data ──
    K_list = sorted(list(set(run["K"] for run in log_data["runs"].values())))
    D_list = sorted(list(set(run["D"] for run in log_data["runs"].values())))
    
    # Build 2D matrices for plotting
    util_matrix = np.zeros((len(D_list), len(K_list)))
    mse_matrix = np.zeros((len(D_list), len(K_list)))
    
    for c_idx, K in enumerate(K_list):
        for r_idx, D in enumerate(D_list):
            run_id = f"K{K}_D{D}"
            if run_id in log_data["runs"]:
                final = log_data["runs"][run_id]["final_metrics"]
                util_matrix[r_idx, c_idx] = final["perplexity"] / K  # perplexity / K
                mse_matrix[r_idx, c_idx] = final["recon_loss"]       # reconstruction MSE

    log2_K = [np.log2(K) for K in K_list]
    
    # =================================================================
    # Image 1: Perplexity / K Heatmap with D = log2(K) line
    # =================================================================
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(util_matrix, origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=1)
    cbar = fig.colorbar(im, label="Normalized Perplexity (Perplexity / K)")
    
    ax.set_xticks(range(len(K_list)))
    ax.set_xticklabels([f"{K}\n(log2={int(np.log2(K))})" for K in K_list])
    ax.set_yticks(range(len(D_list)))
    ax.set_yticklabels(D_list)
    
    ax.set_xlabel("Codebook Size (K)")
    ax.set_ylabel("Latent Dimension (D)")
    ax.set_title("Fig 1: Phase Transition Heatmap (Perplexity/K)\nOverlay with Prediction Line D = log2(K)")
    
    # Add values inside cells
    for r in range(len(D_list)):
        for c in range(len(K_list)):
            ax.text(c, r, f"{util_matrix[r, c]:.2f}", ha="center", va="center", 
                    color="white" if util_matrix[r, c] < 0.6 else "black", fontweight="bold")

    # Superimpose D = log2(K) line
    # Map actual D positions back to y-axis index coordinates using linear interpolation
    y_indices = []
    for K in K_list:
        l2k = np.log2(K)
        # Find index coordinate of l2k in D_list
        idx = np.interp(l2k, D_list, range(len(D_list)))
        y_indices.append(idx)
        
    ax.plot(range(len(K_list)), y_indices, 'r--', marker='o', linewidth=2.5, label=r'Prediction: $D = \log_2 K$')
    ax.legend(loc="upper left")
    
    plt.tight_layout()
    local_fig_path = plot_dir / "fig1_phase_heatmap.png"
    plt.savefig(local_fig_path, dpi=150)
    plt.close()
    sync_file_to_drive(local_fig_path, Path(drive_dir) / "exp1" / "plots" / "fig1_phase_heatmap.png" if drive_dir else None)
    print("  ✓ Figure 1 saved.")

    # =================================================================
    # Image 2: Reconstruction matrix grids for digits 4, 9, 5
    # =================================================================
    # Find target images
    targets = {4: None, 9: None, 5: None}
    for i in range(len(test_dataset)):
        img, label = test_dataset[i]
        if label in targets and targets[label] is None:
            targets[label] = (img, label)
        if all(v is not None for v in targets.values()):
            break
            
    for digit, (raw_img, _) in targets.items():
        fig2, axes = plt.subplots(len(D_list), len(K_list), figsize=(2.5 * len(K_list), 2.5 * len(D_list)))
        fig2.suptitle(f"Reconstruction Matrix for Digit {digit}\n(Cols: K, Rows: D)", fontsize=14, y=0.99)
        
        # Ensure axes is always a 2D array regardless of dimensions
        if len(D_list) == 1 and len(K_list) == 1:
            axes = np.array([[axes]])
        elif len(D_list) == 1:
            axes = np.expand_dims(axes, 0)
        elif len(K_list) == 1:
            axes = np.expand_dims(axes, 1)
            
        for c_idx, K in enumerate(K_list):
            for r_idx, D in enumerate(D_list):
                ax_sub = axes[r_idx, c_idx]
                run_id = f"K{K}_D{D}"
                model_pth = model_dir / f"vqvae_{run_id}.pth"
                
                if model_pth.exists():
                    # Load model and run inference
                    model = VQVAE(latent_dim=D, codebook_K=K)
                    model.load_state_dict(torch.load(model_pth, map_location="cpu"))
                    model.eval()
                    
                    with torch.no_grad():
                        _, _, _, x_recon, _ = model(raw_img.unsqueeze(0))
                        recon_img = ((x_recon.squeeze() + 1) / 2).clamp(0, 1).numpy()
                        
                    ax_sub.imshow(recon_img, cmap='gray')
                else:
                    ax_sub.text(0.5, 0.5, "N/A", ha="center", va="center")
                    
                ax_sub.axis('off')
                if r_idx == 0:
                    ax_sub.set_title(f"K={K}", fontsize=10, fontweight="bold")
                if c_idx == 0:
                    ax_sub.text(-0.2, 0.5, f"D={D}", transform=ax_sub.transAxes, 
                                rotation=0, ha="right", va="center", fontsize=10, fontweight="bold")
                    
        plt.tight_layout()
        local_fig_path = plot_dir / f"fig2_reconstruct_digit{digit}.png"
        plt.savefig(local_fig_path, dpi=150)
        plt.close()
        sync_file_to_drive(local_fig_path, Path(drive_dir) / "exp1" / "plots" / f"fig2_reconstruct_digit{digit}.png" if drive_dir else None)
        print(f"  ✓ Figure 2 (Digit {digit}) saved.")

    # =================================================================
    # Image 3: Data Collapse shifted curves: D - log2(K)
    # =================================================================
    fig3, ax3 = plt.subplots(figsize=(8, 5.5))
    
    for K_idx, K in enumerate(K_list):
        utils = []
        shifted_Ds = []
        l2k = np.log2(K)
        for D in D_list:
            run_id = f"K{K}_D{D}"
            if run_id in log_data["runs"]:
                utils.append(log_data["runs"][run_id]["final_metrics"]["perplexity"] / K)
                shifted_Ds.append(D - l2k)
                
        ax3.plot(shifted_Ds, utils, marker='o', linewidth=2, label=f"K={K} (log2K={int(l2k)})")
        
    ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.6, label='Transition Boundary (D - log2K = 0)')
    ax3.set_xlabel(r'Shifted Latent Dimension ($D - \log_2 K$)')
    ax3.set_ylabel('Normalized Perplexity (Perplexity / K)')
    ax3.set_title("Fig 3: Data Collapse Alignment\nAll curves align when mapped to scaled dimension D - log2(K)")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    local_fig_path = plot_dir / "fig3_data_collapse.png"
    plt.savefig(local_fig_path, dpi=150)
    plt.close()
    sync_file_to_drive(local_fig_path, Path(drive_dir) / "exp1" / "plots" / "fig3_data_collapse.png" if drive_dir else None)
    print("  ✓ Figure 3 saved.")

    # =================================================================
    # Image 4: Phase Boundary Scaling Law (d* vs log2(K))
    # =================================================================
    fig4, ax4 = plt.subplots(figsize=(7, 5))
    d_stars = []
    
    for K in K_list:
        utils = []
        for D in D_list:
            run_id = f"K{K}_D{D}"
            utils.append(log_data["runs"][run_id]["final_metrics"]["perplexity"] / K)
        # Find 50% crossing critical dimension
        d_star = find_crossing(D_list, utils, 0.50)
        d_stars.append(d_star)
        
    # Fit linear scaling law
    slope, intercept = np.polyfit(log2_K, d_stars, 1)
    fit_line = [slope * l2k + intercept for l2k in log2_K]
    
    ax4.scatter(log2_K, d_stars, color='blue', s=100, zorder=5, label='Empirical $d^*(K)$ (50% Perplexity/K)')
    ax4.plot(log2_K, fit_line, 'r-', linewidth=2, label=f'Linear Fit: $d^*(K) = {slope:.2f} \\log_2 K + {intercept:.2f}$')
    
    ax4.set_xlabel(r'$\log_2 K$')
    ax4.set_ylabel(r'Critical Latent Dimension $d^*(K)$')
    ax4.set_title("Fig 4: Phase Boundary Scaling Law\nLinear Fit of Critical Dimensionality vs log2(K)")
    ax4.set_xticks(log2_K)
    ax4.set_xticklabels([f"{int(l2k)}" for l2k in log2_K])
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    local_fig_path = plot_dir / "fig4_scaling_law.png"
    plt.savefig(local_fig_path, dpi=150)
    plt.close()
    sync_file_to_drive(local_fig_path, Path(drive_dir) / "exp1" / "plots" / "fig4_scaling_law.png" if drive_dir else None)
    print("  ✓ Figure 4 saved.")

    # =================================================================
    # Image 5: Reconstruction MSE Heatmap with overlaid contours
    # =================================================================
    fig5, ax5 = plt.subplots(figsize=(7.5, 6))
    im5 = ax5.imshow(mse_matrix, origin='lower', aspect='auto', cmap='plasma')
    cbar5 = fig5.colorbar(im5, label="Reconstruction MSE")
    
    ax5.set_xticks(range(len(K_list)))
    ax5.set_xticklabels([f"{K}" for K in K_list])
    ax5.set_yticks(range(len(D_list)))
    ax5.set_yticklabels(D_list)
    
    ax5.set_xlabel("Codebook Size (K)")
    ax5.set_ylabel("Latent Dimension (D)")
    ax5.set_title("Fig 5: Reconstruction MSE Heatmap & Phase Contours\nOverlay: 50% Utilization Contour & Quality Boundary Contour")
    
    # Label cell values
    for r in range(len(D_list)):
        for c in range(len(K_list)):
            ax5.text(c, r, f"{mse_matrix[r, c]:.4f}", ha="center", va="center", 
                     color="white" if mse_matrix[r, c] < 0.15 else "black", fontweight="bold")

    # Extract boundaries for contours using bilinear mesh if dimensions are large enough
    if util_matrix.shape[0] >= 2 and util_matrix.shape[1] >= 2:
        try:
            K_mesh, D_mesh = np.meshgrid(range(len(K_list)), range(len(D_list)))
            
            # 1. 50% Utilization Contour (Red)
            c1 = ax5.contour(K_mesh, D_mesh, util_matrix, levels=[0.5], colors='red', linewidths=3)
            # 2. Acceptable Reconstruction Quality MSE threshold e.g. 0.13 (Blue)
            c2 = ax5.contour(K_mesh, D_mesh, mse_matrix, levels=[0.13], colors='cyan', linewidths=3)
            
            # Setup custom legends for the contours
            h1, _ = c1.legend_elements()
            h2, _ = c2.legend_elements()
            ax5.legend([h1[0], h2[0]], ["Collapse Boundary (50% Perplexity/K)", "Quality Threshold (MSE = 0.13)"], loc="upper left")
        except Exception as e:
            print(f"  [Plot WARNING] Could not draw contours for Figure 5: {e}")
    else:
        print("  [Plot INFO] Grid too small to draw contour lines for Figure 5. Skipping contours.")
    
    plt.tight_layout()
    local_fig_path = plot_dir / "fig5_tradeoff_contours.png"
    plt.savefig(local_fig_path, dpi=150)
    plt.close()
    sync_file_to_drive(local_fig_path, Path(drive_dir) / "exp1" / "plots" / "fig5_tradeoff_contours.png" if drive_dir else None)
    print("  ✓ Figure 5 saved.")

    # =================================================================
    # Image 6: 每条曲线的 Transition Width (ΔD vs log2(K))
    # =================================================================
    fig6, ax6 = plt.subplots(figsize=(7, 5))
    delta_Ds = []
    
    for K in K_list:
        utils = []
        for D in D_list:
            run_id = f"K{K}_D{D}"
            utils.append(log_data["runs"][run_id]["final_metrics"]["perplexity"] / K)
            
        # D where utilization crosses 80% and 20%
        d_80 = find_crossing(D_list, utils, 0.80)
        d_20 = find_crossing(D_list, utils, 0.20)
        delta_Ds.append(d_20 - d_80)  # Since utilization falls as D increases
        
    ax6.plot(log2_K, delta_Ds, 'go-', linewidth=2.5, markersize=8, label=r'Transition Width $\Delta D = D_{20\%} - D_{80\%}$')
    
    ax6.set_xlabel(r'$\log_2 K$')
    ax6.set_ylabel(r'Transition Width $\Delta D$')
    ax6.set_title("Fig 6: Transition Width vs log2(K)\nWidth of Codebook Collapse Transition Interval")
    ax6.set_xticks(log2_K)
    ax6.set_xticklabels([f"{int(l2k)}" for l2k in log2_K])
    ax6.set_ylim(0, max(delta_Ds) * 1.3)
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    plt.tight_layout()
    local_fig_path = plot_dir / "fig6_transition_width.png"
    plt.savefig(local_fig_path, dpi=150)
    plt.close()
    sync_file_to_drive(local_fig_path, Path(drive_dir) / "exp1" / "plots" / "fig6_transition_width.png" if drive_dir else None)
    print("  ✓ Figure 6 saved.")


def generate_exp2_plots(log_data, plot_dir, drive_dir):
    print("\nGenerating Experiment 2 figures...")
    
    D_list = sorted(list(set(run["D"] for run in log_data["runs"].values())))
    seeds = sorted(list(set(run["seed"] for run in log_data["runs"].values())))
    
    # Collect perplexity arrays per dimension
    perplexity_dict = {D: [] for D in D_list}
    for run in log_data["runs"].values():
        D = run["D"]
        perplexity_dict[D].append(run["final_metrics"]["perplexity"])
        
    # =================================================================
    # Image 7: Strip Plot of final perplexity distributions
    # =================================================================
    fig7, ax7 = plt.subplots(figsize=(7, 5))
    
    rng = np.random.default_rng(42)
    
    for idx, D in enumerate(D_list):
        perps = perplexity_dict[D]
        # Jitter to avoid overlapping dots horizontally
        jitter = rng.uniform(-0.1, 0.1, size=len(perps))
        ax7.scatter(np.full_like(perps, D) + jitter, perps, alpha=0.8, 
                    edgecolors='navy', c='skyblue', s=60, label=f"D={D}" if idx == 0 else "")
                    
    ax7.set_xticks(D_list)
    ax7.set_xticklabels([f"D={D}" for D in D_list])
    ax7.set_xlabel("Latent Dimension (D)")
    ax7.set_ylabel("Final Epoch Codebook Perplexity")
    ax7.set_title("Fig 7: Final Perplexity Distribution Across 10 Seeds\n(D = 8, 9, 10)")
    ax7.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    local_fig_path = plot_dir / "fig7_strip_plot.png"
    plt.savefig(local_fig_path, dpi=150)
    plt.close()
    sync_file_to_drive(local_fig_path, Path(drive_dir) / "exp2" / "plots" / "fig7_strip_plot.png" if drive_dir else None)
    print("  ✓ Figure 7 saved.")

    # =================================================================
    # Image 8: Training dynamics (Perplexity vs epochs) for D=9
    # =================================================================
    fig8, ax8 = plt.subplots(figsize=(8, 5.5))
    
    # Gather D=9 runs
    cmap = plt.get_cmap("tab10")
    color_idx = 0
    epochs = []
    
    for run_id, run in log_data["runs"].items():
        if run["D"] == 9:
            epochs = [metrics["epoch"] for metrics in run["epoch_metrics"]]
            perps = [metrics["perplexity"] for metrics in run["epoch_metrics"]]
            seed = run["seed"]
            ax8.plot(epochs, perps, marker='x', alpha=0.8, color=cmap(color_idx % 10), 
                     linewidth=1.8, label=f"Seed {seed}")
            color_idx += 1
            
    ax8.set_xlabel("Training Epoch")
    ax8.set_ylabel("Perplexity")
    ax8.set_title("Fig 8: VQ-VAE Training Dynamics (Latent Dimension D = 9)\nPerplexity Trajectories Across 10 Seeds")
    ax8.grid(True, alpha=0.3)
    
    if len(epochs) > 0:
        ax8.set_xticks(epochs)
        ax8.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    else:
        ax8.text(0.5, 0.5, "No runs with D=9 available", ha="center", va="center", fontsize=12)
    
    plt.tight_layout()
    local_fig_path = plot_dir / "fig8_dynamics_D9.png"
    plt.savefig(local_fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    sync_file_to_drive(local_fig_path, Path(drive_dir) / "exp2" / "plots" / "fig8_dynamics_D9.png" if drive_dir else None)
    print("  ✓ Figure 8 saved.")

    # =================================================================
    # Image 9: Across-seed Mean ± Std comparison error bar plot
    # =================================================================
    fig9, ax9 = plt.subplots(figsize=(7, 5))
    
    means = []
    stds = []
    
    for D in D_list:
        perps = perplexity_dict[D]
        means.append(np.mean(perps))
        stds.append(np.std(perps))
        
    ax9.errorbar(D_list, means, yerr=stds, fmt='o-', color='crimson', ecolor='navy',
                 linewidth=2.5, elinewidth=2, capsize=6, capthick=2, markersize=8, label=r'Perplexity ($\mu \pm \sigma$)')
                 
    ax9.set_xticks(D_list)
    ax9.set_xticklabels([f"D={D}" for D in D_list])
    ax9.set_xlabel("Latent Dimension (D)")
    ax9.set_ylabel("Across-Seed Final Perplexity")
    ax9.set_title("Fig 9: Across-Seed Perplexity Comparison\nMean ± Std of Codebook Perplexity for D = 8, 9, 10")
    ax9.grid(True, alpha=0.3)
    ax9.legend()
    
    plt.tight_layout()
    local_fig_path = plot_dir / "fig9_error_bar.png"
    plt.savefig(local_fig_path, dpi=150)
    plt.close()
    sync_file_to_drive(local_fig_path, Path(drive_dir) / "exp2" / "plots" / "fig9_error_bar.png" if drive_dir else None)
    print("  ✓ Figure 9 saved.")


# =====================================================================
# 5. Main Execution Block
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="VQ-VAE Collapse Phase Transition Experiments")
    parser.add_argument("--drive-dir", type=str, default=None, 
                        help="Optional Google Drive output folder (e.g. /content/drive/MyDrive/vqvae_collapse)")
    parser.add_argument("--epochs1", type=int, default=45, help="Number of epochs for Experiment 1 (default 45)")
    parser.add_argument("--epochs2", type=int, default=45, help="Number of epochs for Experiment 2 (default 45)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training (default 128)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for Experiment 1 (default 42)")
    parser.add_argument("--workers", type=int, default=2, help="Number of parallel experiment workers (default 2)")
    parser.add_argument("--dry-run", action="store_true", help="Perform a quick dry run with minimal epochs/grids")
    args = parser.parse_args()
    
    # 1. Device and Dataset Pre-download
    device = get_device()
    device_name = str(device)
    print(f"\n[Hardware Accelerator]: Detected device = {device_name}")
    enable_tf32(device)
    
    # Pre-download dataset in parent process to avoid worker download race conditions
    print("[Dataset Setup]: Ensuring MNIST dataset is fully downloaded locally...")
    get_mnist(train=True, download=True)
    test_dataset = get_mnist(train=False, download=True)
    
    print(f"[Dataset Setup]: Loaded full test set size: {len(test_dataset)}")
    
    # Determine local outputs directory
    output_dir = ROOT / "outputs" / "collapse_experiment"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Run Experiment 1 (Grid of K x D)
    run_experiment_1(args, device_name, test_dataset, output_dir, args.drive_dir)
    
    # 3. Run Experiment 2 (Seed Stochasticity)
    run_experiment_2(args, device_name, output_dir, args.drive_dir)
    
    print("\n" + "="*80)
    print("  ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"  Local outputs directory: {output_dir}")
    if args.drive_dir:
        print(f"  Google Drive synced directory: {args.drive_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        # Crucial for GPU/MPS safety inside parallel spawned processes
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
