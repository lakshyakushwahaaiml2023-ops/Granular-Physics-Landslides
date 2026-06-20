import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from models.unet2d import UNet2D, RunoutPredictor, GranularPhysicsLoss
from data.generator import LandslideDataset
from core.particles import ParticleState
from core.terrain import straight_slope
from core.dem_solver import DEMSolver


class GranularTrainer:
    """
    Manages the sequential training and validation of the UNet2D dynamics model 
    and the RunoutPredictor CNN regressor.
    """
    def __init__(self, device: str, checkpoint_dir='checkpoints/'):
        self.device = torch.device(device)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Paths for H5 datasets
        self.train_h5 = 'data/train.h5'
        self.val_h5 = 'data/val.h5'
        
        # Best model filenames
        self.dynamics_ckpt = os.path.join(self.checkpoint_dir, 'dynamics_best.pt')
        self.runout_ckpt = os.path.join(self.checkpoint_dir, 'runout_best.pt')

    def train_dynamics_model(self, epochs=40, batch_size=16, lr=1e-3, preload=False):
        """
        Trains the UNet2D dynamics model using GranularPhysicsLoss.
        """
        train_dataset = LandslideDataset(self.train_h5, mode='next_state', preload=preload)
        val_dataset = LandslideDataset(self.val_h5, mode='next_state', preload=preload)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        model = UNet2D().to(self.device)
        criterion = GranularPhysicsLoss()
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_val_loss = float('inf')
        
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            
            # Tracking metrics
            total_mass_err = 0.0
            total_momentum_err = 0.0
            samples_count = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                pred = model(batch_x)
                losses = criterion(pred, batch_y, batch_x)
                loss = losses['total']
                loss.backward()
                optimizer.step()
                
                # Update loss accumulators
                bs = batch_x.size(0)
                epoch_loss += loss.item() * bs
                samples_count += bs
                
                # Compute training metrics for logging
                with torch.no_grad():
                    pred_mass = pred[:, 0, :, :].sum(dim=(-1, -2))
                    target_mass = batch_y[:, 0, :, :].sum(dim=(-1, -2))
                    mass_err = torch.abs(pred_mass - target_mass) / (target_mass + 1e-12)
                    total_mass_err += mass_err.sum().item()
                    
                    pred_px = (pred[:, 0] * pred[:, 1]).sum(dim=(-1, -2))
                    target_px = (batch_y[:, 0] * batch_y[:, 1]).sum(dim=(-1, -2))
                    momentum_err = torch.abs(pred_px - target_px) / (torch.abs(target_px) + 1e-12)
                    total_momentum_err += momentum_err.sum().item()
            
            scheduler.step()
            
            epoch_loss /= samples_count
            epoch_mass_err = total_mass_err / samples_count
            epoch_momentum_err = total_momentum_err / samples_count
            
            # Print epoch logs
            print(f"Epoch {epoch:2d} | loss: {epoch_loss:.4f} | mass_err: {epoch_mass_err:.6f} | momentum_err: {epoch_momentum_err:.6f}")
            
            # Validation Step
            model.eval()
            val_loss = 0.0
            val_samples = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    pred = model(batch_x)
                    losses = criterion(pred, batch_y, batch_x)
                    val_loss += losses['total'].item() * batch_x.size(0)
                    val_samples += batch_x.size(0)
            val_loss /= val_samples
            
            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), self.dynamics_ckpt)
            
            # Clean up open files in dataset
            train_dataset.close()
            val_dataset.close()

    def train_runout_model(self, epochs=30, batch_size=32, lr=5e-4, preload=True):
        """
        Trains the RunoutPredictor CNN using Huber Loss.
        """
        train_dataset = LandslideDataset(self.train_h5, mode='runout', preload=preload)
        val_dataset = LandslideDataset(self.val_h5, mode='runout', preload=preload)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        model = RunoutPredictor().to(self.device)
        optimizer = AdamW(model.parameters(), lr=lr)
        
        best_val_loss = float('inf')
        
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            
            # Tracking metrics
            total_ae = 0.0
            total_re = 0.0
            samples_count = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                pred = model(batch_x)
                loss = F.huber_loss(pred, batch_y, delta=0.2)
                loss.backward()
                optimizer.step()
                
                bs = batch_x.size(0)
                epoch_loss += loss.item() * bs
                samples_count += bs
                
                with torch.no_grad():
                    ae = torch.abs(pred - batch_y)
                    re = ae / (batch_y + 1e-12)
                    total_ae += ae.sum().item()
                    total_re += re.sum().item()
                    
            epoch_loss /= samples_count
            epoch_mae = total_ae / samples_count
            epoch_relative_err = (total_re / samples_count) * 100.0
            
            # Print epoch logs
            print(f"Epoch {epoch:2d} | MAE: {epoch_mae:.3f} m | relative_err: {epoch_relative_err:.1f}%")
            
            # Validation Step
            model.eval()
            val_loss = 0.0
            val_samples = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    pred = model(batch_x)
                    loss = F.huber_loss(pred, batch_y, delta=0.2)
                    val_loss += loss.item() * batch_x.size(0)
                    val_samples += batch_x.size(0)
            val_loss /= val_samples
            
            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), self.runout_ckpt)

        train_dataset.close()
        val_dataset.close()

    def evaluate_both(self):
        """
        Performs full model validation evaluations, measuring accuracy, conservation laws, and speedups.
        """
        print("\n--- Model Evaluation Results ---")
        
        # 1. Load checkpoints
        dynamics_model = UNet2D().to(self.device)
        dynamics_model.load_state_dict(torch.load(self.dynamics_ckpt, map_location=self.device))
        dynamics_model.eval()
        
        runout_model = RunoutPredictor().to(self.device)
        runout_model.load_state_dict(torch.load(self.runout_ckpt, map_location=self.device))
        runout_model.eval()
        
        val_next_state = LandslideDataset(self.val_h5, mode='next_state')
        loader_dynamics = DataLoader(val_next_state, batch_size=1, shuffle=False)
        
        val_runout = LandslideDataset(self.val_h5, mode='runout')
        loader_runout = DataLoader(val_runout, batch_size=1, shuffle=False)
        
        # --- Dynamics model evaluation ---
        total_mass_err = 0.0
        total_mae = 0.0
        n_dyn = len(val_next_state)
        
        # Measure U-Net Inference time
        t_start = time.time()
        with torch.no_grad():
            for batch_x, batch_y in loader_dynamics:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred = dynamics_model(batch_x)
                
                # Mass error
                pred_mass = pred[0, 0].sum().item()
                target_mass = batch_y[0, 0].sum().item()
                total_mass_err += abs(pred_mass - target_mass) / (target_mass + 1e-12)
                
                # Grid MAE
                total_mae += F.l1_loss(pred, batch_y).item()
                
        unet_duration = (time.time() - t_start) / n_dyn  # average time per step
        
        mean_mass_err = (total_mass_err / n_dyn) * 100.0
        mean_mae = total_mae / n_dyn
        
        # Measure DEM Solver time for 20 steps (the skip parameter)
        # We can construct a typical evaluation particle pile and slope
        terrain = straight_slope(angle_deg=35.0)
        state = ParticleState.from_random_pile(N=200, pile_x=0.5, pile_y=2.0, pile_width=0.4, pile_height=0.3)
        solver = DEMSolver(dt=5e-5)
        
        # Warmup solver
        _ = solver.step_n(state, terrain, 20)
        
        # Measure
        t0 = time.time()
        for _ in range(50):
            _ = solver.step_n(state, terrain, 20)
        dem_duration = (time.time() - t0) / 50.0  # average time for 20 DEM steps
        
        # Calculate speedup
        speedup = dem_duration / unet_duration if unet_duration > 0 else 0.0
        
        print("Dynamics model:")
        print(f"  Mean mass conservation error: {mean_mass_err:.1f}%")
        print(f"  Mean MAE on grid: {mean_mae:.4f}")
        print(f"  Speedup vs DEM: {speedup:.1f}x")
        
        # --- Runout predictor evaluation ---
        errors = []
        rel_errors = []
        n_run = len(val_runout)
        
        t_start = time.time()
        with torch.no_grad():
            for batch_x, batch_y in loader_runout:
                batch_x = batch_x.to(self.device)
                pred = runout_model(batch_x)
                pred_val = pred.item()
                target_val = batch_y.item()
                
                err = abs(pred_val - target_val)
                errors.append(err)
                rel_errors.append(err / (target_val + 1e-12))
                
        runout_duration_ms = ((time.time() - t_start) / n_run) * 1000.0  # average inference time in ms
        
        mean_mae_m = np.mean(errors)
        mean_re_pct = np.mean(rel_errors) * 100.0
        worst_case_m = np.max(errors)
        
        print("Runout predictor:")
        print(f"  Mean absolute error: {mean_mae_m:.3f} m")
        print(f"  Mean relative error: {mean_re_pct:.1f}%")
        print(f"  Worst-case error: {worst_case_m:.3f} m")
        print(f"  Inference time: {runout_duration_ms:.1f} ms (single sample)")

        val_next_state.close()
        val_runout.close()


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = GranularTrainer(device=device)
    print("Training dynamics model...")
    trainer.train_dynamics_model(epochs=40)
    print("\nTraining runout predictor...")
    trainer.train_runout_model(epochs=30)
    trainer.evaluate_both()
