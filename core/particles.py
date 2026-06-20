import numpy as np

class ParticleState:
    """
    Manages the state of N particles in the DEM simulation.
    """
    def __init__(self, pos, vel, radius, mass, active=None):
        self.pos = np.asarray(pos, dtype=np.float64)       # (N, 2)
        self.vel = np.asarray(vel, dtype=np.float64)       # (N, 2)
        self.radius = np.asarray(radius, dtype=np.float64) # (N,)
        self.mass = np.asarray(mass, dtype=np.float64)     # (N,)
        if active is None:
            self.active = np.ones(len(self.radius), dtype=bool)
        else:
            self.active = np.asarray(active, dtype=bool)   # (N,)

    @property
    def N(self) -> int:
        return len(self.radius)

    @classmethod
    def from_random_pile(cls, N: int, pile_x: float, pile_y: float, pile_width: float, pile_height: float) -> 'ParticleState':
        """
        Pack N particles into a rectangular pile using random non-overlapping placement (rejection sampling).
        """
        pos_list = []
        radius_list = []
        mass_list = []
        density = 2650.0  # kg/m^3 (granite)

        for i in range(N):
            # Polydisperse: radius uniform in [0.008, 0.015]
            r = np.random.uniform(0.008, 0.015)
            
            # Define bounding box for center placement
            x_min = pile_x + r if pile_width >= 2 * r else pile_x
            x_max = pile_x + pile_width - r if pile_width >= 2 * r else pile_x + pile_width
            y_min = pile_y + r if pile_height >= 2 * r else pile_y
            y_max = pile_y + pile_height - r if pile_height >= 2 * r else pile_y + pile_height
            
            success = False
            for attempt in range(1000):
                px = np.random.uniform(x_min, x_max)
                py = np.random.uniform(y_min, y_max)
                
                # Check overlap with existing particles
                overlap = False
                for j in range(len(pos_list)):
                    dx = px - pos_list[j][0]
                    dy = py - pos_list[j][1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < (r + radius_list[j]):
                        overlap = True
                        break
                
                if not overlap:
                    pos_list.append([px, py])
                    radius_list.append(r)
                    mass = density * np.pi * (r ** 2)
                    mass_list.append(mass)
                    success = True
                    break
            
            if not success:
                # Print a warning and stop generating more particles
                print(f"Warning: Rejection sampling failed to place particle {i+1}/{N} after 1000 attempts. "
                      f"Generated {len(pos_list)} particles.")
                break

        n_placed = len(pos_list)
        pos = np.array(pos_list, dtype=np.float64) if n_placed > 0 else np.zeros((0, 2))
        vel = np.zeros((n_placed, 2), dtype=np.float64)
        radius = np.array(radius_list, dtype=np.float64)
        mass = np.array(mass_list, dtype=np.float64)
        active = np.ones(n_placed, dtype=bool)

        return cls(pos, vel, radius, mass, active)

    def kinetic_energy(self) -> float:
        """
        Translational kinetic energy of all active particles: KE = 0.5 * sum(m * v^2)
        """
        if self.N == 0:
            return 0.0
        v_sq = np.sum(self.vel[self.active] ** 2, axis=1)
        return float(0.5 * np.sum(self.mass[self.active] * v_sq))

    def potential_energy(self, g: float = 9.81) -> float:
        """
        Potential energy of all active particles: PE = sum(m * g * y)
        """
        if self.N == 0:
            return 0.0
        return float(np.sum(self.mass[self.active] * g * self.pos[self.active, 1]))

    def total_energy(self, g: float = 9.81) -> float:
        return self.kinetic_energy() + self.potential_energy(g)

    def copy(self) -> 'ParticleState':
        return ParticleState(
            self.pos.copy(),
            self.vel.copy(),
            self.radius.copy(),
            self.mass.copy(),
            self.active.copy()
        )
