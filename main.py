import os
import time
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Import project modules
from core.particles import ParticleState
from core.terrain import concave_slope, Terrain
from core.dem_solver import DEMSolver
from models.unet2d import UNet2D, RunoutPredictor, GranularPhysicsLoss
from data.generator import LandslideGenerator, LandslideDataset, grid_to_visualization, particles_to_grid
from utils.metrics import runout_distance, deposit_centroid, max_deposit_height, flow_front_velocity, mobility_ratio

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)


def train_next_state(train_path: str, val_path: str, epochs=10, batch_size=32, lr=1e-3, device='cpu'):
    """
    Trains the 2D U-Net for next-state prediction.
    """
    print("\n--- Training U-Net for Next-State Prediction ---")
    train_dataset = LandslideDataset(train_path, mode='next_state')
    val_dataset = LandslideDataset(val_path, mode='next_state')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = UNet2D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = GranularPhysicsLoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            pred = model(batch_x)
            losses = criterion(pred, batch_y, batch_x)
            loss = losses['total']
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_x.size(0)
            
        epoch_loss /= len(train_dataset)
        train_losses.append(epoch_loss)
        
        # Validation
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x)
                losses = criterion(pred, batch_y, batch_x)
                loss = losses['total']
                epoch_val_loss += loss.item() * batch_x.size(0)
        epoch_val_loss /= len(val_dataset)
        val_losses.append(epoch_val_loss)
        
        print(f"Epoch {epoch}/{epochs} | Train Loss: {epoch_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")
        
    return model, train_losses, val_losses, val_dataset


def train_runout_predictor(train_path: str, val_path: str, epochs=10, batch_size=16, lr=1e-3, device='cpu'):
    """
    Trains the CNN regressor for runout distance prediction from initial state.
    """
    print("\n--- Training CNN for Final Runout Distance Prediction ---")
    train_dataset = LandslideDataset(train_path, mode='runout')
    val_dataset = LandslideDataset(val_path, mode='runout')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = RunoutPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.HuberLoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_x.size(0)
            
        epoch_loss /= len(train_dataset)
        train_losses.append(epoch_loss)
        
        # Validation
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                epoch_val_loss += loss.item() * batch_x.size(0)
        epoch_val_loss /= len(val_dataset)
        val_losses.append(epoch_val_loss)
        
        print(f"Epoch {epoch}/{epochs} | Train Loss: {epoch_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        
    return model, train_losses, val_losses, val_dataset


def run_dem_visualization(artifact_dir: str):
    """
    Runs a single DEM landslide simulation and plots the dynamic states over time.
    """
    print("\nRunning a DEM landslide simulation for visualization...")
    terrain = concave_slope(angle_top=42.0, angle_bottom=15.0, length=4.0)
    solver = DEMSolver(dt=5e-5)
    
    state = ParticleState.from_random_pile(
        N=220, pile_x=0.5, pile_y=1.9, pile_width=0.4, pile_height=0.4
    )
    
    times = [0.0]
    states = [state.copy()]
    
    dt_physical = solver.dt
    curr_state = state
    for i in range(1, 3001):
        curr_state = solver.step(curr_state, terrain)
        if i % 600 == 0:
            times.append(i * dt_physical)
            states.append(curr_state.copy())
            
    print("Simulation complete. Generating visualization...")
    
    # Generate scatter plot colored by velocity magnitude
    fig, axes = plt.subplots(len(states), 1, figsize=(10, 12), sharex=True)
    fig.suptitle("Granular Landslide Simulation (DEM Model)", fontsize=16, color="#2c3e50", fontweight='bold')
    
    for idx, (t, s) in enumerate(zip(times, states)):
        ax = axes[idx]
        ax.set_facecolor("#fafafa")
        ax.grid(True, linestyle="--", alpha=0.5)
        
        # Plot terrain
        ax.plot(terrain.points[:, 0], terrain.points[:, 1], color="#7f8c8d", linewidth=3, label="Slope profile")
        ax.fill_between(terrain.points[:, 0], terrain.points[:, 1], 0, color="#d5dbdb", alpha=0.5)
        
        # Scatter active particles colored by velocity
        active = s.active
        if np.sum(active) > 0:
            v_mags = np.sqrt(np.sum(s.vel[active]**2, axis=1))
            sc = ax.scatter(s.pos[active, 0], s.pos[active, 1], s=s.radius[active]*1200, 
                            c=v_mags, cmap="coolwarm", edgecolor="k", linewidths=0.5, zorder=5, vmin=0, vmax=2.5)
            if idx == 0:
                cbar = fig.colorbar(sc, ax=axes, orientation="vertical", shrink=0.6, pad=0.02)
                cbar.set_label("Velocity (m/s)", fontsize=12)
                
        # Draw settled particles
        inactive = ~active
        if np.sum(inactive) > 0:
            ax.scatter(s.pos[inactive, 0], s.pos[inactive, 1], s=s.radius[inactive]*1200, 
                       color="#bdc3c7", edgecolor="#7f8c8d", linewidths=0.5, alpha=0.7, zorder=4)
                       
        ax.set_xlim(0, 4.0)
        ax.set_ylim(0, 3.0)
        ax.set_ylabel("Elevation y (m)", fontsize=10)
        ax.text(0.1, 2.7, f"Time: {t:.3f} s (KE: {s.kinetic_energy():.4f} J)", 
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="#bdc3c7"))
                
    axes[-1].set_xlabel("Horizontal Position x (m)", fontsize=12)
    
    dem_plot_path_art = os.path.join(artifact_dir, "dem_landslide_simulation.png")
    dem_plot_path_ws = "dem_landslide_simulation.png"
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(dem_plot_path_art, dpi=150)
    plt.savefig(dem_plot_path_ws, dpi=150)
    plt.close()
    
    # Compute metrics
    runout = runout_distance(states[-1], release_x=0.5)
    centroid = deposit_centroid(states[-1])
    max_h = max_deposit_height(states[-1], terrain)
    vels = flow_front_velocity(states, times)
    mobility = mobility_ratio(states[-1], release_height=2.5 - 0.2, runout_dist=runout)
    
    print("\n--- Landslide Metrics ---")
    print(f"Runout Distance: {runout:.3f} m")
    print(f"Deposit Centroid: ({centroid[0]:.3f}, {centroid[1]:.3f})")
    print(f"Max Deposit Height: {max_h:.3f} m")
    print(f"Peak Flow Front Velocity: {np.max(vels) if len(vels) > 0 else 0.0:.3f} m/s")
    print(f"Mobility Ratio (H/L): {mobility:.3f}")
    print("-------------------------\n")


def plot_unet_predictions(model, val_dataset, device, artifact_dir: str):
    """
    Evaluates the U-Net on a validation sample and generates a comparative grid-based visualization.
    """
    model.eval()
    
    sample_idx = np.random.randint(0, len(val_dataset))
    batch_x, batch_y = val_dataset[sample_idx]
    
    # Add batch dim
    batch_x_gpu = batch_x.unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred = model(batch_x_gpu).cpu().squeeze(0).numpy() # (4, 64, 128)
        
    x_np = batch_x.numpy()  # (5, 64, 128)
    y_np = batch_y.numpy()  # (4, 64, 128)
    
    # Extract channels
    density_gt = y_np[0]
    density_pred = pred[0]
    
    ke_gt = y_np[3]
    ke_pred = pred[3]
    
    terrain_mask = x_np[4]
    
    # Mask out areas inside terrain for clean visualization
    density_gt[terrain_mask > 0.5] = np.nan
    density_pred[terrain_mask > 0.5] = np.nan
    ke_gt[terrain_mask > 0.5] = np.nan
    ke_pred[terrain_mask > 0.5] = np.nan
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle("PADL U-Net Prediction vs. DEM Ground Truth", fontsize=16, fontweight='bold', color="#2c3e50")
    
    def setup_ax(ax, title, data, cmap="viridis", vmin=None, vmax=None):
        ax.set_facecolor("#2c3e50") # Dark background
        im = ax.imshow(data, origin="lower", cmap=cmap, extent=[0, 4, 0, 3], vmin=vmin, vmax=vmax)
        
        # Draw terrain surface line
        x_grid = np.linspace(0, 4, 128)
        terrain_y = []
        for col_idx in range(128):
            terrain_rows = np.where(terrain_mask[:, col_idx] > 0.5)[0]
            if len(terrain_rows) > 0:
                y_val = (terrain_rows[-1] / 64.0) * 3.0
            else:
                y_val = 0.2
            terrain_y.append(y_val)
        ax.plot(x_grid, terrain_y, color="#e74c3c", linewidth=2.0, linestyle="--")
        ax.set_title(title, fontsize=12)
        ax.set_xlim(0, 4.0)
        ax.set_ylim(0, 3.0)
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
        
    setup_ax(axes[0, 0], "Density (DEM Ground Truth)", density_gt, vmin=0, vmax=12.0)
    setup_ax(axes[0, 1], "Density (U-Net Prediction)", density_pred, vmin=0, vmax=12.0)
    
    setup_ax(axes[1, 0], "Kinetic Energy (DEM Ground Truth)", ke_gt, cmap="magma", vmin=0, vmax=0.05)
    setup_ax(axes[1, 1], "Kinetic Energy (U-Net Prediction)", ke_pred, cmap="magma", vmin=0, vmax=0.05)
    
    for ax in axes.flat:
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        
    plt.tight_layout()
    comparison_plot_path_art = os.path.join(artifact_dir, "landslide_unet_comparison.png")
    comparison_plot_path_ws = "landslide_unet_comparison.png"
    plt.savefig(comparison_plot_path_art, dpi=150)
    plt.savefig(comparison_plot_path_ws, dpi=150)
    plt.close()
    print("U-Net predictions plotted and saved successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="2D Granular Landslide Simulator & PADL Acceleration Pipeline")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    args = parser.parse_args()
    
    artifact_dir = r"C:\Users\PREDATOR\.gemini\antigravity\brain\1447afcd-0fde-4a9a-8a98-b5a8f3019e5e"
    train_h5 = "data/train.h5"
    val_h5 = "data/val.h5"
    
    # 1. Run DEM simulation for physics visualization
    run_dem_visualization(artifact_dir)
    
    # 2. Wait or check if h5 files exist (they are being generated by the background task)
    print("\nChecking for training datasets...")
    for _ in range(30):
        if os.path.exists(train_h5) and os.path.exists(val_h5):
            # Check file size is stable (not writing anymore)
            time.sleep(2)
            break
        print("Waiting for data/generator.py background dataset task to finish...")
        time.sleep(10)
        
    if not (os.path.exists(train_h5) and os.path.exists(val_h5)):
        # Fallback to generating a tiny dataset locally if background task hasn't finished
        print("Datasets not found. Generating a small subset for training locally...")
        generator = LandslideGenerator(skip=20)
        generator.generate_dataset(n_trajectories=15, save_path=train_h5)
        generator.generate_dataset(n_trajectories=4, save_path=val_h5)
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 3. Train U-Net for Next State prediction
    unet_model, u_train_losses, u_val_losses, u_val_ds = train_next_state(
        train_path=train_h5,
        val_path=val_h5,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=1e-3,
        device=device
    )
    
    # Plot learning curves for U-Net
    plt.figure(figsize=(8, 5))
    plt.plot(u_train_losses, label="Train Loss", color="#1abc9c", linewidth=2)
    plt.plot(u_val_losses, label="Validation Loss", color="#e67e22", linewidth=2)
    plt.title("U-Net Next-State Prediction Learning Curves", fontsize=12, fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(artifact_dir, "unet_learning_curves.png"), dpi=150)
    plt.savefig("unet_learning_curves.png", dpi=150)
    plt.close()
    
    # Evaluate and compare next state predictions
    plot_unet_predictions(unet_model, u_val_ds, device, artifact_dir)
    
    # 4. Train Runout Predictor
    runout_model, r_train_losses, r_val_losses, r_val_ds = train_runout_predictor(
        train_path=train_h5,
        val_path=val_h5,
        epochs=args.epochs,
        batch_size=16,
        lr=1e-3,
        device=device
    )
    
    # Plot learning curves for Runout predictor
    plt.figure(figsize=(8, 5))
    plt.plot(r_train_losses, label="Train Loss", color="#2980b9", linewidth=2)
    plt.plot(r_val_losses, label="Validation Loss", color="#d35400", linewidth=2)
    plt.title("Runout Predictor Learning Curves", fontsize=12, fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (m^2)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(artifact_dir, "runout_learning_curves.png"), dpi=150)
    plt.savefig("runout_learning_curves.png", dpi=150)
    plt.close()
    
    # Print some comparisons
    runout_model.eval()
    print("\n--- Runout Predictor Sample Val Evaluations ---")
    val_loader = DataLoader(r_val_ds, batch_size=5, shuffle=True)
    for batch_x, batch_y in val_loader:
        batch_x = batch_x.to(device)
        with torch.no_grad():
            preds = runout_model(batch_x).cpu().numpy()
        gts = batch_y.numpy()
        for idx in range(len(preds)):
            print(f"Sample {idx+1}: Predicted Runout = {preds[idx]:.3f} m | Ground-truth Runout = {gts[idx]:.3f} m")
        break
        
    print("\nPipeline execution complete! All plots saved to the workspace and artifacts folder.")
