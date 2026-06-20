import numpy as np
from core.particles import ParticleState

def runout_distance(state: ParticleState, release_x: float) -> float:
    """
    Returns the horizontal distance from release_x to the furthest active particle.
    If no particles are active, uses the furthest particle overall.
    """
    if state.N == 0:
        return 0.0
    active_mask = state.active
    if np.sum(active_mask) > 0:
        max_x = np.max(state.pos[active_mask, 0])
    else:
        max_x = np.max(state.pos[:, 0])
    return max(0.0, float(max_x - release_x))

def deposit_centroid(state: ParticleState) -> tuple[float, float]:
    """
    Returns the center of mass of the settled deposit (or all particles if none are active).
    """
    if state.N == 0:
        return (0.0, 0.0)
    
    # Settled means inactive (or low velocity, but here we check state.active == False or state.active == True)
    # The prompt says "center of mass of settled deposit".
    # Active particles are still flowing, inactive particles have settled or left.
    # Let's find the centroid of the inactive particles that are still within domain.
    # Wait, what if we use the inactive particles?
    inactive_mask = ~state.active
    if np.sum(inactive_mask) > 0:
        pos_settled = state.pos[inactive_mask]
        mass_settled = state.mass[inactive_mask]
        total_mass = np.sum(mass_settled)
        if total_mass > 1e-12:
            cx = np.sum(pos_settled[:, 0] * mass_settled) / total_mass
            cy = np.sum(pos_settled[:, 1] * mass_settled) / total_mass
            return (float(cx), float(cy))
            
    # Fallback to all particles if none are settled
    total_mass = np.sum(state.mass)
    if total_mass > 1e-12:
        cx = np.sum(state.pos[:, 0] * state.mass) / total_mass
        cy = np.sum(state.pos[:, 1] * state.mass) / total_mass
        return (float(cx), float(cy))
    return (0.0, 0.0)

def max_deposit_height(state: ParticleState, terrain, x_bins: int = 50) -> float:
    """
    Returns the peak deposit thickness above the terrain surface.
    """
    if state.N == 0:
        return 0.0
        
    # We look at settled (inactive) particles, or all particles within domain.
    # Let's check particles that are still in the domain (y > 0.0).
    pos = state.pos
    radius = state.radius
    
    # We bin x from min(pos_x) to max(pos_x) or over the terrain domain (0, 4)
    x_min, x_max = 0.0, 4.0
    bins = np.linspace(x_min, x_max, x_bins + 1)
    
    bin_max_heights = np.zeros(x_bins)
    
    for i in range(state.N):
        px, py = pos[i]
        r = radius[i]
        
        # Find which bin this particle center falls into
        if px < x_min or px > x_max:
            continue
        bin_idx = int((px - x_min) / (x_max - x_min) * x_bins)
        bin_idx = min(x_bins - 1, max(0, bin_idx))
        
        # Get terrain elevation at this particle's x
        signed_dist, normal = terrain.distance_to_surface([px, py])
        # The terrain point directly below/near is C = pos - signed_dist * normal
        # Height above terrain is simply the top of the particle (py + r) minus the terrain surface y.
        # To get the terrain surface y at px, let's find the point C where signed_dist is projected.
        C_y = py - signed_dist * normal[1]
        thickness = (py + r) - C_y
        
        if thickness > bin_max_heights[bin_idx]:
            bin_max_heights[bin_idx] = thickness
            
    return float(np.max(bin_max_heights))

def flow_front_velocity(states: list[ParticleState], times: list[float]) -> np.ndarray:
    """
    Calculates the velocity of the leading edge over time from a sequence of states.
    Returns an array of length len(states) - 1.
    """
    if len(states) < 2:
        return np.zeros(0)
        
    front_positions = []
    for state in states:
        active_mask = state.active
        if np.sum(active_mask) > 0:
            front_x = np.max(state.pos[active_mask, 0])
        else:
            front_x = np.max(state.pos[:, 0])
        front_positions.append(front_x)
        
    velocities = []
    for k in range(len(states) - 1):
        dt_val = times[k+1] - times[k]
        if dt_val > 1e-12:
            vel = (front_positions[k+1] - front_positions[k]) / dt_val
        else:
            vel = 0.0
        velocities.append(vel)
        
    return np.array(velocities, dtype=np.float64)

def mobility_ratio(state: ParticleState, release_height: float, runout_dist: float) -> float:
    """
    H/L ratio: standard geotechnical measure of landslide mobility.
    """
    if runout_dist <= 1e-6:
        return 0.0
    return float(release_height / runout_dist)
