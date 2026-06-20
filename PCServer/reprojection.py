import numpy as np
import math

def get_depth_at_rgb_pixel(u_rgb, v_rgb, depth_map, rgb_K, depth_K, R_d2r, T_d2r):
    # 0. Safety Guard
    if rgb_K[0, 0] == 0.0 or rgb_K[1, 1] == 0.0:
        return None

    h_d, w_d = depth_map.shape
    
    # 1. We need the inverse transform: RGB to Depth
    R_r2d = R_d2r.T
    T_r2d = -R_r2d @ T_d2r

    # Precompute the precise direction of the ray in RGB space (D_rgb)
    dx = (u_rgb - rgb_K[0, 2]) / rgb_K[0, 0]
    dy = (v_rgb - rgb_K[1, 2]) / rgb_K[1, 1]
    D_rgb = np.array([[dx], [dy], [1.0]])

    # Precompute the rotation and translation Z-components to handle lens tilt
    R_z_row = R_r2d[2, :] 
    z_dot = float(R_z_row @ D_rgb)
    T_z = float(T_r2d[2, 0])

    # 2. Initial Guess
    Z_guess = 1.0 
    
    for _ in range(5):
        # Step A: 3D point using current Z_guess
        P_rgb = D_rgb * Z_guess

        # Step B: Transform into Depth camera space
        P_depth = R_r2d @ P_rgb + T_r2d

        if P_depth[2, 0] <= 1e-5 or math.isnan(P_depth[2, 0]):
            return None

        # Step C: Project onto 2D depth map
        u_d_float = (P_depth[0, 0] * depth_K[0, 0] / P_depth[2, 0]) + depth_K[0, 2]
        v_d_float = (P_depth[1, 0] * depth_K[1, 1] / P_depth[2, 0]) + depth_K[1, 2]

        if math.isnan(u_d_float) or math.isnan(v_d_float) or math.isinf(u_d_float) or math.isinf(v_d_float):
            return None

        u_d = int(round(u_d_float))
        v_d = int(round(v_d_float))

        if u_d < 0 or u_d >= w_d or v_d < 0 or v_d >= h_d:
            return None 

        # Step D: Read the ACTUAL depth at that pixel
        Z_actual = depth_map[v_d, u_d]

        if Z_actual <= 0.0 or math.isnan(Z_actual):
            return None 

        # Step E: Convert the Depth-space Z back to an RGB-space Z
        # THIS IS THE FIX: Scales the ray length to account for the physical downward tilt
        new_Z_guess = (float(Z_actual) - T_z) / z_dot

        if abs(new_Z_guess - Z_guess) < 0.005:
            return new_Z_guess

        Z_guess = new_Z_guess

    return Z_guess