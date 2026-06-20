import numpy as np
import h5py
import os
import numba
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from core.particles import ParticleState
from core.terrain import straight_slope, concave_slope, slope_with_obstacle, Terrain
from core.dem_solver import DEMSolver
from utils.metrics import runout_distance


@numba.njit
def compute_terrain_occupancy_jit(terrain_points, H, W, domain_x, domain_y):
    """
    JIT-compiled terrain occupancy grid computation.
    Evaluates signed distance to polyline for each grid cell.
    """
    occupancy = np.zeros((H, W), dtype=np.float32)
    dx = (domain_x[1] - domain_x[0]) / W
    dy = (domain_y[1] - domain_y[0]) / H
    M = len(terrain_points)
    
    for r in range(H):
        y_val = (r + 0.5) * dy + domain_y[0]
        for c in range(W):
            x_val = (c + 0.5) * dx + domain_x[0]
            
            best_dist = 1e9
            best_signed_dist = 0.0
            
            for j in range(M - 1):
                P1 = terrain_points[j]
                P2 = terrain_points[j+1]
                
                vx = P2[0] - P1[0]
                vy = P2[1] - P1[1]
                v_len_sq = vx*vx + vy*vy
                if v_len_sq < 1e-12:
                    continue
                    
                wx = x_val - P1[0]
                wy = y_val - P1[1]
                t = (wx*vx + wy*vy) / v_len_sq
                t_clamped = max(0.0, min(1.0, t))
                
                Cx = P1[0] + t_clamped * vx
                Cy = P1[1] + t_clamped * vy
                
                dx_val = x_val - Cx
                dy_val = y_val - Cy
                dist = np.sqrt(dx_val*dx_val + dy_val*dy_val)
                
                if dist < best_dist:
                    best_dist = dist
                    n_seg_x = -vy
                    n_seg_y = vx
                    n_seg_len = np.sqrt(vx*vx + vy*vy)
                    n_seg_x /= n_seg_len
                    n_seg_y /= n_seg_len
                    best_signed_dist = dx_val*n_seg_x + dy_val*n_seg_y
                    
            if best_signed_dist <= 0:
                occupancy[r, c] = 1.0
                
    return occupancy


@numba.njit
def particles_to_grid_jit(pos, vel, radius, mass, active, domain_x, domain_y, H, W):
    """
    JIT-compiled bilinear splatting for grid conversion.
    Distributes particle quantities to the 4 nearest grid cell centers.
    """
    dx = (domain_x[1] - domain_x[0]) / W
    dy = (domain_y[1] - domain_y[0]) / H
    cell_area = dx * dy
    
    density_grid = np.zeros((H, W), dtype=np.float64)
    mass_accum = np.zeros((H, W), dtype=np.float64)
    vx_accum = np.zeros((H, W), dtype=np.float64)
    vy_accum = np.zeros((H, W), dtype=np.float64)
    ke_accum = np.zeros((H, W), dtype=np.float64)
    count_accum = np.zeros((H, W), dtype=np.float64)
    
    N = len(pos)
    for i in range(N):
        if not active[i]:
            continue
        
        px, py = pos[i]
        vx, vy = vel[i]
        m = mass[i]
        ke = 0.5 * m * (vx*vx + vy*vy)
        
        c_float = (px - domain_x[0]) / dx - 0.5
        r_float = (py - domain_y[0]) / dy - 0.5
        
        c0 = int(np.floor(c_float))
        c1 = c0 + 1
        r0 = int(np.floor(r_float))
        r1 = r0 + 1
        
        wc1 = c_float - c0
        wc0 = 1.0 - wc1
        wr1 = r_float - r0
        wr0 = 1.0 - wr1
        
        corners = [
            (r0, c0, wr0 * wc0),
            (r0, c1, wr0 * wc1),
            (r1, c0, wr1 * wc0),
            (r1, c1, wr1 * wc1)
        ]
        
        for r, c, w in corners:
            if 0 <= r < H and 0 <= c < W:
                density_grid[r, c] += w
                mass_accum[r, c] += w * m
                vx_accum[r, c] += w * m * vx
                vy_accum[r, c] += w * m * vy
                ke_accum[r, c] += w * ke
                count_accum[r, c] += w
                
    grid = np.zeros((4, H, W), dtype=np.float32)
    grid[0] = (density_grid / cell_area).astype(np.float32)
    
    for r in range(H):
        for c in range(W):
            m_acc = mass_accum[r, c]
            if m_acc > 1e-12:
                grid[1, r, c] = vx_accum[r, c] / m_acc
                grid[2, r, c] = vy_accum[r, c] / m_acc
            
            cnt_acc = count_accum[r, c]
            if cnt_acc > 1e-12:
                grid[3, r, c] = ke_accum[r, c] / cnt_acc
                
    return grid


def particles_to_grid(state: ParticleState, domain_x, domain_y, H=64, W=128) -> np.ndarray:
    """
    Converts a ParticleState to a grid-based representation using JIT bilinear splatting.
    """
    return particles_to_grid_jit(
        state.pos, state.vel, state.radius, state.mass, state.active,
        np.array(domain_x, dtype=np.float64), np.array(domain_y, dtype=np.float64), H, W
    )


def grid_to_visualization(grid: np.ndarray, terrain: Terrain) -> np.ndarray:
    """
    Visualizes the density channel as a heatmap (viridis colormap) and overlays the terrain in white.
    Returns a (H, W, 3) uint8 RGB array.
    """
    H, W = grid.shape[1], grid.shape[2]
    density = grid[0]
    
    d_max = np.max(density)
    if d_max > 1e-6:
        norm_density = np.clip(density / d_max, 0.0, 1.0)
    else:
        norm_density = np.zeros_like(density)
        
    cmap = plt.get_cmap('viridis')
    rgba = cmap(norm_density)  # (H, W, 4)
    rgb = (rgba[:, :, :3] * 255.0).astype(np.uint8)
    
    dx = 4.0 / W
    dy = 3.0 / H
    x_centers = (np.arange(W) + 0.5) * dx
    y_vals = np.interp(x_centers, terrain.points[:, 0], terrain.points[:, 1])
    
    for c in range(W):
        r_idx = int(y_vals[c] / dy)
        if 0 <= r_idx < H:
            rgb[r_idx, c] = [255, 255, 255]
            
    return rgb


class LandslideGenerator:
    """
    Generates dataset trajectories by running DEM simulations and mapping states to grids.
    """
    def __init__(self, H=64, W=128, skip=20, device='cpu'):
        self.H = H
        self.W = W
        self.skip = skip
        self.device = device
        self.domain_x = (0.0, 4.0)
        self.domain_y = (0.0, 3.0)

    def generate_trajectory(self, state0: ParticleState, terrain: Terrain, n_steps=3000, pile_x=0.5) -> dict:
        """
        Runs DEM for n_steps, saving inputs (5, H, W) and targets (4, H, W) every `skip` steps.
        """
        # 1. Compute terrain occupancy grid (1, H, W) using JIT-accelerated helper
        terrain_occupancy = np.zeros((1, self.H, self.W), dtype=np.float32)
        terrain_occupancy[0] = compute_terrain_occupancy_jit(
            terrain.points, self.H, self.W,
            np.array(self.domain_x, dtype=np.float64),
            np.array(self.domain_y, dtype=np.float64)
        )
                    
        # 2. Run simulation and record grids
        solver = DEMSolver(dt=5e-5, domain_x=self.domain_x, domain_y=self.domain_y)
        
        num_frames = n_steps // self.skip
        inputs = np.zeros((num_frames, 5, self.H, self.W), dtype=np.float32)
        targets = np.zeros((num_frames, 4, self.H, self.W), dtype=np.float32)
        
        curr_state = state0
        
        for f in range(num_frames):
            grid_t = particles_to_grid(curr_state, self.domain_x, self.domain_y, self.H, self.W)
            inputs[f, :4] = grid_t
            inputs[f, 4] = terrain_occupancy[0]
            
            # Step solver skip times to get state at t + skip*dt
            next_state = solver.step_n(curr_state, terrain, self.skip)
            
            grid_tp1 = particles_to_grid(next_state, self.domain_x, self.domain_y, self.H, self.W)
            targets[f] = grid_tp1
            
            curr_state = next_state
            
        final_runout = runout_distance(curr_state, release_x=pile_x)
        
        return {
            'inputs': inputs,
            'targets': targets,
            'terrain': terrain_occupancy,
            'metadata': {
                'N_particles': state0.N,
                'final_runout': final_runout
            }
        }

    def generate_dataset(self, n_trajectories=300, save_path='data/train.h5'):
        """
        Runs DEM trajectories with randomized parameters and saves datasets in HDF5 format.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        inputs_list = []
        targets_list = []
        runout_distances = []
        
        slope_angles = []
        n_particles = []
        final_runouts = []
        
        print(f"Generating {n_trajectories} trajectories into {save_path}...")
        for t_idx in tqdm(range(n_trajectories)):
            # Randomize simulation parameters
            slope_angle = np.random.uniform(25.0, 45.0)
            terrain_type_roll = np.random.random()
            
            if terrain_type_roll < 0.5:
                terrain = straight_slope(angle_deg=slope_angle)
            elif terrain_type_roll < 0.8:
                terrain = concave_slope(angle_top=slope_angle, angle_bottom=np.random.uniform(10.0, 20.0))
            else:
                obs_x = np.random.uniform(1.8, 2.6)
                obs_h = np.random.uniform(0.15, 0.35)
                terrain = slope_with_obstacle(angle_deg=slope_angle, obstacle_x=obs_x, obstacle_height=obs_h)
                
            N = int(np.random.randint(150, 401))
            pile_x = np.random.uniform(0.3, 0.8)
            pile_width = np.random.uniform(0.2, 0.5)
            pile_height = np.random.uniform(0.2, 0.4)
            
            # Pack particles
            state0 = ParticleState.from_random_pile(
                N=N, pile_x=pile_x, pile_y=2.0, pile_width=pile_width, pile_height=pile_height
            )
            
            # Run simulation
            traj = self.generate_trajectory(state0, terrain, n_steps=3000, pile_x=pile_x)
            
            inputs_list.append(traj['inputs'])
            targets_list.append(traj['targets'])
            runout_distances.append(traj['metadata']['final_runout'])
            
            slope_angles.append(slope_angle)
            n_particles.append(state0.N)
            final_runouts.append(traj['metadata']['final_runout'])
            
        # Combine
        X = np.concatenate(inputs_list, axis=0) # (n_traj * frames, 5, H, W)
        Y = np.concatenate(targets_list, axis=0) # (n_traj * frames, 4, H, W)
        runout_distances = np.array(runout_distances, dtype=np.float32)
        
        # Save to H5
        with h5py.File(save_path, 'w') as f:
            f.create_dataset('inputs', data=X, compression='gzip')
            f.create_dataset('targets', data=Y, compression='gzip')
            f.create_dataset('runout_distances', data=runout_distances, compression='gzip')
            
            # Metadata group
            meta = f.create_group('metadata')
            meta.create_dataset('slope_angles', data=np.array(slope_angles, dtype=np.float32))
            meta.create_dataset('n_particles', data=np.array(n_particles, dtype=np.int32))
            meta.create_dataset('final_runouts', data=np.array(final_runouts, dtype=np.float32))
            
        print(f"Saved dataset: {save_path}")


class LandslideDataset(torch.utils.data.Dataset):
    """
    Standard PyTorch Dataset for landslide state grids or runout distance targets.
    """
    def __init__(self, h5_path: str, mode='next_state'):
        self.h5_path = h5_path
        self.mode = mode
        
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"Dataset file not found at {h5_path}")
            
        with h5py.File(self.h5_path, 'r') as f:
            self.total_frames = len(f['inputs'])
            self.num_traj = len(f['runout_distances'])
            self.frames_per_traj = self.total_frames // self.num_traj

    def __len__(self):
        if self.mode == 'runout':
            return self.num_traj
        return self.total_frames

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            if self.mode == 'runout':
                frame_idx = idx * self.frames_per_traj
                inp = torch.from_numpy(f['inputs'][frame_idx]).float()
                target = torch.tensor(f['runout_distances'][idx]).float()
                return inp, target
            else:
                inp = torch.from_numpy(f['inputs'][idx]).float()
                target = torch.from_numpy(f['targets'][idx]).float()
                return inp, target


if __name__ == '__main__':
    # Initialize generator
    gen = LandslideGenerator(skip=20)
    
    print("Generating train.h5...")
    gen.generate_dataset(n_trajectories=200, save_path='data/train.h5')
    print("Generating val.h5...")
    gen.generate_dataset(n_trajectories=40,  save_path='data/val.h5')
    
    ds = LandslideDataset('data/train.h5', mode='next_state')
    print(f"Next-state dataset: {len(ds)} samples")
    
    ds2 = LandslideDataset('data/train.h5', mode='runout')
    print(f"Runout prediction dataset: {len(ds2)} samples")
    
    inp, tgt = ds[0]
    print(f"Input grid shape:  {inp.shape}")
    print(f"Target grid shape: {tgt.shape}")
