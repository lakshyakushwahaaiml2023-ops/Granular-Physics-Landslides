import numpy as np

class Terrain:
    """
    Represents the terrain boundary as a 2D polyline.
    """
    def __init__(self, points):
        # points is list of (x, y) coordinates, convert to shape (M, 2)
        self.points = np.asarray(points, dtype=np.float64)
        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError("Terrain points must be of shape (M, 2)")
        if len(self.points) < 2:
            raise ValueError("Terrain must have at least 2 points")

    def distance_to_surface(self, point) -> tuple[float, np.ndarray]:
        """
        Returns the signed distance and the outward normal vector at the nearest point on the terrain surface.
        - signed_distance: positive if the point is above the terrain, negative if below.
        - normal_vec: (2,) unit normal vector pointing up/outwards from the terrain surface.
        """
        Q = np.asarray(point, dtype=np.float64)
        best_dist = float('inf')
        best_normal = np.array([0.0, 1.0])
        best_signed_dist = 0.0

        M = len(self.points)
        for j in range(M - 1):
            P1 = self.points[j]
            P2 = self.points[j+1]
            
            # Segment vector and projection
            v = P2 - P1
            w = Q - P1
            v_len_sq = np.dot(v, v)
            if v_len_sq < 1e-12:
                continue
                
            t = np.dot(w, v) / v_len_sq
            t_clamped = max(0.0, min(1.0, t))
            C = P1 + t_clamped * v
            
            # Distance from Q to closest point C on this segment
            diff = Q - C
            dist = np.sqrt(np.dot(diff, diff))
            
            if dist < best_dist:
                best_dist = dist
                
                # Outward normal for this segment (orthogonal pointing left/up of travel direction P1 -> P2)
                # For travel in +x, (dx, dy) has normal (-dy, dx)
                dx, dy = v[0], v[1]
                n_seg = np.array([-dy, dx], dtype=np.float64)
                n_seg_len = np.sqrt(dx*dx + dy*dy)
                n_seg /= n_seg_len
                
                # Signed distance is dot product of diff with normal
                signed_dist = np.dot(diff, n_seg)
                best_signed_dist = signed_dist
                best_normal = n_seg

        return best_signed_dist, best_normal


def straight_slope(angle_deg: float, length: float = 4.0) -> Terrain:
    """
    Simple inclined plane at given angle transitioning to a flat runout zone.
    """
    x = np.linspace(0.0, length, 100)
    # Start elevation around 2.5m, slope down to a flat floor at y = 0.2m
    angle_rad = np.radians(angle_deg)
    y = np.maximum(0.2, 2.5 - x * np.tan(angle_rad))
    points = np.column_stack((x, y))
    return Terrain(points)


def concave_slope(angle_top: float, angle_bottom: float, length: float = 4.0) -> Terrain:
    """
    Steep upper section transitioning smoothly to a gentler runout zone.
    """
    x = np.linspace(0.0, length, 100)
    y = np.zeros_like(x)
    y[0] = 2.5
    dx = length / (len(x) - 1)
    
    for i in range(len(x) - 1):
        t = x[i] / length
        # Linear interpolation of slope angle
        angle = np.radians(angle_top * (1.0 - t) + angle_bottom * t)
        y[i+1] = y[i] - dx * np.tan(angle)
        
    y = np.maximum(0.2, y)
    points = np.column_stack((x, y))
    return Terrain(points)


def slope_with_obstacle(angle_deg: float, obstacle_x: float, obstacle_height: float, length: float = 4.0) -> Terrain:
    """
    Slope with a smooth bump (obstacle/deflector) that deflects granular flow.
    """
    x = np.linspace(0.0, length, 150)
    y = np.maximum(0.2, 2.5 - x * np.tan(np.radians(angle_deg)))
    
    # Gaussian bump obstacle for numerical stability in contact physics
    width = 0.15
    bump = obstacle_height * np.exp(-((x - obstacle_x) / width) ** 2)
    y += bump
    
    points = np.column_stack((x, y))
    return Terrain(points)


def valley_cross_section(width: float, depth: float) -> Terrain:
    """
    V-shaped valley cross-section: debris channeled between two slopes.
    """
    x = np.linspace(0.0, width, 100)
    # y ranges from depth down to 0.2m at the center (x = width/2)
    y = np.maximum(0.2, 0.2 + (depth - 0.2) * (2.0 * np.abs(x - width / 2.0) / width))
    points = np.column_stack((x, y))
    return Terrain(points)
