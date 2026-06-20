import numpy as np
import numba
import time
from core.particles import ParticleState
from core.terrain import Terrain, straight_slope
from utils.metrics import runout_distance

@numba.njit
def build_spatial_hash(pos, active, cell_size, nx, ny, domain_x_min, domain_y_min):
    """
    Builds a spatial hash grid for particle index lookup in O(N) time.
    """
    N = len(pos)
    num_cells = nx * ny
    cell_idx = np.zeros(N, dtype=numba.int32)
    cell_counts = np.zeros(num_cells, dtype=numba.int32)
    
    # First pass: count particles per cell
    for i in range(N):
        if not active[i]:
            cell_idx[i] = -1
            continue
        cx = int((pos[i, 0] - domain_x_min) / cell_size)
        cy = int((pos[i, 1] - domain_y_min) / cell_size)
        
        # Clamp to grid boundaries
        if cx < 0: cx = 0
        if cx >= nx: cx = nx - 1
        if cy < 0: cy = 0
        if cy >= ny: cy = ny - 1
        
        c_idx = cx * ny + cy
        cell_idx[i] = c_idx
        cell_counts[c_idx] += 1
        
    # Cumulative sum to compute offsets
    cell_offsets = np.zeros(num_cells + 1, dtype=numba.int32)
    running_sum = 0
    for c in range(num_cells):
        cell_offsets[c] = running_sum
        running_sum += cell_counts[c]
    cell_offsets[num_cells] = running_sum
    
    # Populate particle indices in cell bins
    particle_indices = np.zeros(N, dtype=numba.int32)
    current_offsets = cell_offsets.copy()[:-1]
    
    for i in range(N):
        if not active[i]:
            continue
        c_idx = cell_idx[i]
        particle_indices[current_offsets[c_idx]] = i
        current_offsets[c_idx] += 1
        
    return cell_offsets, particle_indices, cell_idx

@numba.njit
def compute_forces(pos, vel, radius, mass, active, terrain_points, E, nu, mu_s, e_n, dt,
                   cell_size, nx, ny, domain_x_min, domain_y_min,
                   cell_offsets, particle_indices, cell_idx):
    """
    Computes contact forces (particle-particle and particle-terrain) using Hertz-Mindlin physics.
    """
    N = len(pos)
    forces = np.zeros((N, 2), dtype=numba.float64)
    
    # 1. Particle-Particle contacts
    for i in range(N):
        if not active[i]:
            continue
        
        cx = int((pos[i, 0] - domain_x_min) / cell_size)
        cy = int((pos[i, 1] - domain_y_min) / cell_size)
        
        # Search 9 neighboring cells
        for dx in range(-1, 2):
            nx_cell = cx + dx
            if nx_cell < 0 or nx_cell >= nx:
                continue
            for dy in range(-1, 2):
                ny_cell = cy + dy
                if ny_cell < 0 or ny_cell >= ny:
                    continue
                
                c_idx = nx_cell * ny + ny_cell
                start_p = cell_offsets[c_idx]
                end_p = cell_offsets[c_idx + 1]
                
                for p_idx in range(start_p, end_p):
                    j = particle_indices[p_idx]
                    # Enforce j > i to avoid self-collisions and double counting
                    if j <= i or not active[j]:
                        continue
                    
                    diff_x = pos[j, 0] - pos[i, 0]
                    diff_y = pos[j, 1] - pos[i, 1]
                    dist = np.sqrt(diff_x*diff_x + diff_y*diff_y)
                    r_sum = radius[i] + radius[j]
                    
                    if dist < r_sum and dist > 1e-12:
                        delta = r_sum - dist
                        n_x = diff_x / dist
                        n_y = diff_y / dist
                        
                        v_rel_x = vel[j, 0] - vel[i, 0]
                        v_rel_y = vel[j, 1] - vel[i, 1]
                        
                        v_n = v_rel_x * n_x + v_rel_y * n_y
                        v_t_x = v_rel_x - v_n * n_x
                        v_t_y = v_rel_y - v_n * n_y
                        
                        E_eff = E / (2.0 * (1.0 - nu*nu))
                        R_eff = (radius[i] * radius[j]) / r_sum
                        mass_eff = (mass[i] * mass[j]) / (mass[i] + mass[j])
                        
                        k_n = (4.0 / 3.0) * E_eff * np.sqrt(R_eff * delta)
                        ln_e = np.log(e_n)
                        c_n = 2.0 * np.sqrt(mass_eff * k_n) * (-ln_e / np.sqrt(np.pi*np.pi + ln_e*ln_e))
                        
                        # Normal force (spring-dashpot, repulsive only)
                        F_n_mag = k_n * delta - c_n * v_n
                        if F_n_mag < 0.0:
                            F_n_mag = 0.0
                        
                        F_n_x = F_n_mag * n_x
                        F_n_y = F_n_mag * n_y
                        
                        # Tangential force (Coulomb friction limited)
                        k_t = 0.8 * k_n
                        F_t_trial_x = -k_t * v_t_x * dt
                        F_t_trial_y = -k_t * v_t_y * dt
                        F_t_trial_norm = np.sqrt(F_t_trial_x*F_t_trial_x + F_t_trial_y*F_t_trial_y)
                        
                        limit = mu_s * F_n_mag
                        if F_t_trial_norm <= limit:
                            F_t_x = F_t_trial_x
                            F_t_y = F_t_trial_y
                        else:
                            if F_t_trial_norm > 1e-12:
                                F_t_x = limit * F_t_trial_x / F_t_trial_norm
                                F_t_y = limit * F_t_trial_y / F_t_trial_norm
                            else:
                                F_t_x = 0.0
                                F_t_y = 0.0
                                
                        forces[i, 0] += F_n_x + F_t_x
                        forces[i, 1] += F_n_y + F_t_y
                        forces[j, 0] -= F_n_x + F_t_x
                        forces[j, 1] -= F_n_y + F_t_y
                        
        # 2. Particle-Terrain contacts
        best_dist = 1e9
        best_signed_dist = 0.0
        best_n_x = 0.0
        best_n_y = 0.0
        
        M = len(terrain_points)
        for j in range(M - 1):
            P1 = terrain_points[j]
            P2 = terrain_points[j+1]
            
            vx = P2[0] - P1[0]
            vy = P2[1] - P1[1]
            v_len_sq = vx*vx + vy*vy
            if v_len_sq < 1e-12:
                continue
                
            wx = pos[i, 0] - P1[0]
            wy = pos[i, 1] - P1[1]
            t = (wx*vx + wy*vy) / v_len_sq
            t_clamped = max(0.0, min(1.0, t))
            
            Cx = P1[0] + t_clamped * vx
            Cy = P1[1] + t_clamped * vy
            
            dx = pos[i, 0] - Cx
            dy = pos[i, 1] - Cy
            dist = np.sqrt(dx*dx + dy*dy)
            
            if dist < best_dist:
                best_dist = dist
                
                # Outward segment normal
                n_seg_x = -vy
                n_seg_y = vx
                n_seg_len = np.sqrt(vx*vx + vy*vy)
                n_seg_x /= n_seg_len
                n_seg_y /= n_seg_len
                
                best_signed_dist = dx*n_seg_x + dy*n_seg_y
                best_n_x = n_seg_x
                best_n_y = n_seg_y
                
        if best_signed_dist < radius[i]:
            delta = radius[i] - best_signed_dist
            n_x = best_n_x
            n_y = best_n_y
            
            v_rel_x = 0.0 - vel[i, 0]
            v_rel_y = 0.0 - vel[i, 1]
            
            v_n = v_rel_x * n_x + v_rel_y * n_y
            v_t_x = v_rel_x - v_n * n_x
            v_t_y = v_rel_y - v_n * n_y
            
            # Terrain contact physics: infinite mass and infinite radius
            E_eff = E / (2.0 * (1.0 - nu*nu))
            R_eff = radius[i]
            mass_eff = mass[i]
            
            k_n = (4.0 / 3.0) * E_eff * np.sqrt(R_eff * delta)
            ln_e = np.log(e_n)
            c_n = 2.0 * np.sqrt(mass_eff * k_n) * (-ln_e / np.sqrt(np.pi*np.pi + ln_e*ln_e))
            
            F_n_mag = k_n * delta - c_n * v_n
            if F_n_mag < 0.0:
                F_n_mag = 0.0
                
            F_n_x = F_n_mag * n_x
            F_n_y = F_n_mag * n_y
            
            k_t = 0.8 * k_n
            F_t_trial_x = -k_t * v_t_x * dt
            F_t_trial_y = -k_t * v_t_y * dt
            F_t_trial_norm = np.sqrt(F_t_trial_x*F_t_trial_x + F_t_trial_y*F_t_trial_y)
            
            limit = mu_s * F_n_mag
            if F_t_trial_norm <= limit:
                F_t_x = F_t_trial_x
                F_t_y = F_t_trial_y
            else:
                if F_t_trial_norm > 1e-12:
                    F_t_x = limit * F_t_trial_x / F_t_trial_norm
                    F_t_y = limit * F_t_trial_y / F_t_trial_norm
                else:
                    F_t_x = 0.0
                    F_t_y = 0.0
                    
            forces[i, 0] += F_n_x + F_t_x
            forces[i, 1] += F_n_y + F_t_y
            
    return forces

@numba.njit
def verlet_step1(pos, vel, mass, active, forces, dt):
    """
    First part of Velocity-Verlet: update velocity by half step and position by full step.
    """
    N = len(pos)
    g_vec = np.array([0.0, -9.81], dtype=numba.float64)
    for i in range(N):
        if not active[i]:
            continue
        vel[i, 0] += 0.5 * (forces[i, 0] / mass[i] + g_vec[0]) * dt
        vel[i, 1] += 0.5 * (forces[i, 1] / mass[i] + g_vec[1]) * dt
        
        pos[i, 0] += vel[i, 0] * dt
        pos[i, 1] += vel[i, 1] * dt

@numba.njit
def verlet_step2(vel, mass, active, forces, dt):
    """
    Second part of Velocity-Verlet: update velocity by half step using recomputed forces.
    """
    N = len(vel)
    g_vec = np.array([0.0, -9.81], dtype=numba.float64)
    for i in range(N):
        if not active[i]:
            continue
        vel[i, 0] += 0.5 * (forces[i, 0] / mass[i] + g_vec[0]) * dt
        vel[i, 1] += 0.5 * (forces[i, 1] / mass[i] + g_vec[1]) * dt

@numba.njit
def check_domain_boundaries(pos, active, domain_x, domain_y):
    """
    Deactivates particles that settle outside the physical domain bounds.
    """
    N = len(pos)
    for i in range(N):
        if not active[i]:
            continue
        if (pos[i, 0] < domain_x[0] or pos[i, 0] > domain_x[1] or
            pos[i, 1] < domain_y[0] or pos[i, 1] > domain_y[1]):
            active[i] = False


class DEMSolver:
    """
    Runs the 2D DEM granular simulation with JIT compiled contact models and integrators.
    """
    def __init__(self, dt=5e-5, domain_x=(0, 4), domain_y=(0, 3)):
        self.dt = dt
        self.domain_x = np.array(domain_x, dtype=np.float64)
        self.domain_y = np.array(domain_y, dtype=np.float64)
        
        # Physical parameters (SI units)
        self.E = 1e7          # Young's modulus
        self.nu = 0.3         # Poisson's ratio
        self.mu_s = 0.5       # static friction coefficient
        self.mu_r = 0.1       # rolling friction coefficient (stored for configuration)
        self.e_n = 0.6        # normal restitution coefficient
        
    def step(self, state: ParticleState, terrain: Terrain) -> ParticleState:
        """
        Runs one step of the DEM physics engine. Returns a new ParticleState.
        """
        new_state = state.copy()
        if new_state.N == 0:
            return new_state
            
        # 1. Spatial Hash Parameters
        max_r = np.max(new_state.radius)
        cell_size = 3.0 * max_r
        nx = int(np.ceil((self.domain_x[1] - self.domain_x[0]) / cell_size))
        ny = int(np.ceil((self.domain_y[1] - self.domain_y[0]) / cell_size))
        
        # 2. Build Hash Grid
        cell_offsets, particle_indices, cell_idx = build_spatial_hash(
            new_state.pos, new_state.active, cell_size, nx, ny, self.domain_x[0], self.domain_y[0]
        )
        
        # 3. Initial Forces
        forces = compute_forces(
            new_state.pos, new_state.vel, new_state.radius, new_state.mass, new_state.active,
            terrain.points, self.E, self.nu, self.mu_s, self.e_n, self.dt,
            cell_size, nx, ny, self.domain_x[0], self.domain_y[0],
            cell_offsets, particle_indices, cell_idx
        )
        
        # 4. Integrate Step 1
        verlet_step1(
            new_state.pos, new_state.vel, new_state.mass, new_state.active, forces, self.dt
        )
        
        # 5. Boundary Check
        check_domain_boundaries(new_state.pos, new_state.active, self.domain_x, self.domain_y)
        
        # 6. Recompute Hash Grid and Forces at the new positions
        cell_offsets, particle_indices, cell_idx = build_spatial_hash(
            new_state.pos, new_state.active, cell_size, nx, ny, self.domain_x[0], self.domain_y[0]
        )
        
        new_forces = compute_forces(
            new_state.pos, new_state.vel, new_state.radius, new_state.mass, new_state.active,
            terrain.points, self.E, self.nu, self.mu_s, self.e_n, self.dt,
            cell_size, nx, ny, self.domain_x[0], self.domain_y[0],
            cell_offsets, particle_indices, cell_idx
        )
        
        # 7. Integrate Step 2
        verlet_step2(
            new_state.vel, new_state.mass, new_state.active, new_forces, self.dt
        )
        
        return new_state

    def step_n(self, state: ParticleState, terrain: Terrain, n: int) -> ParticleState:
        """
        Runs n steps of the DEM physics engine, returns the final state.
        """
        curr_state = state
        for _ in range(n):
            curr_state = self.step(curr_state, terrain)
        return curr_state

    def is_settled(self, state: ParticleState, prev_state: ParticleState = None, tol=1e-4) -> bool:
        """
        Returns True if the maximum velocity of active particles drops below a tolerance threshold.
        """
        active_mask = state.active
        if np.sum(active_mask) == 0:
            return True
        v_mags = np.sqrt(np.sum(state.vel[active_mask]**2, axis=1))
        return float(np.max(v_mags)) < tol


if __name__ == '__main__':
    # Initialize slope and particle pile
    terrain = straight_slope(angle_deg=35, length=4.0)
    state = ParticleState.from_random_pile(N=200, 
              pile_x=0.5, pile_y=2.0, 
              pile_width=0.4, pile_height=0.3)
    solver = DEMSolver(dt=5e-5)
    
    # Warm up JIT compilers
    print("Compiling JIT functions...")
    warmup_start = time.time()
    _ = solver.step(state, terrain)
    print(f"JIT Warmup complete in {time.time() - warmup_start:.2f}s")
    
    t0 = time.time()
    for i in range(500):
        state = solver.step(state, terrain)
    dt_wall = time.time() - t0
    
    print(f"200 particles, 500 steps: {dt_wall:.2f}s")
    print(f"Steps/sec: {500/dt_wall:.0f}")
    print(f"Final KE: {state.kinetic_energy():.4f} J")
    print(f"Runout so far: {runout_distance(state, release_x=0.5):.3f} m")
