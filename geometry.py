import numpy as np

def get_uv(P, A, B, C):
    """
    Returns (u,v) where P_proj = A + u*(B-A) + v*(C-A)
    Supports: [3], [N, 3], and [3, H, W]
    """
    is_image = (P.ndim == 3 and P.shape[0] == 3 and P.shape[1] != P.shape[0])
    if is_image:
        P = np.transpose(P, (1, 2, 0))
    curr_shape = P.shape
    P_flat = P.reshape(-1, 3) if P.ndim > 1 else np.expand_dims(P, 0)
    
    A, B, C = A.reshape(1, 3).astype(float), B.reshape(1, 3).astype(float), C.reshape(1, 3).astype(float)
    
    v0 = B - A
    v1 = C - A
    v2 = P_flat - A
    
    d00 = np.sum(v0 * v0, axis=-1)
    d01 = np.sum(v0 * v1, axis=-1)
    d11 = np.sum(v1 * v1, axis=-1)
    d20 = np.sum(v2 * v0, axis=-1)
    d21 = np.sum(v2 * v1, axis=-1)
    
    denom = d00 * d11 - d01 * d01 + 1e-10
    
    u = (d11 * d20 - d01 * d21) / denom
    v = (d00 * d21 - d01 * d20) / denom
    
    uv = np.stack([u, v], axis=-1)
    
    uv = uv.reshape(*curr_shape[:-1], 2)
    if is_image:
        uv = np.transpose(uv, (2, 0, 1)) # [2, H, W]
    
    return uv

def uv_to_3d(uv, A, B, C):
    """
    Converts (u, v) coordinates back to 3D space: P = A + u*(B-A) + v*(C-A)
    uv supports shape [..., 2] or [2, H, W]
    """
    is_image = (uv.ndim == 3 and uv.shape[0] == 2 and uv.shape[1] != uv.shape[0])
    
    A, B, C = A.reshape(-1, 3).astype(float), B.reshape(-1, 3).astype(float), C.reshape(-1, 3).astype(float)
    
    if is_image:
        u = uv[0:1, ...]
        v = uv[1:2, ...]
        A_val = A.reshape(3, 1, 1)
        B_val = B.reshape(3, 1, 1)
        C_val = C.reshape(3, 1, 1)
    else:
        u = uv[..., 0:1]
        v = uv[..., 1:2]
        A_val = A.reshape(1, 3) # Or broadcast natively
        B_val = B.reshape(1, 3)
        C_val = C.reshape(1, 3)
        
    return A_val + u * (B_val - A_val) + v * (C_val - A_val)

def project_to_plane(P, A, B, C):
    """
    Projects points P onto the infinite plane defined by A, B, C using UV space.
    """
    uv = get_uv(P, A, B, C)
    res = uv_to_3d(uv, A, B, C)
    return np.clip(res, 0, 255).astype(P.dtype)

def mask_plane_labels(uv, v_O, t_Ap, t_Bp, t_Cp):
    """
    Returns tensor of same spatial shape as uv, with values 1, 2, or 3 based on angular sectors.
    Used for both Plane and Triangle classification modes.
    Assumes uv is already generated via get_uv() or project_to_triangle().
    """
    is_image = (uv.ndim == 3 and uv.shape[0] == 2 and uv.shape[1] != uv.shape[0])
    
    if is_image:
        uv = np.transpose(uv, (1, 2, 0))
    curr_shape = uv.shape
    uv_flat = uv.reshape(-1, 2)
    
    u_o = v_O[1]
    v_o = v_O[2]
    pO = np.array([u_o, v_o], dtype=uv.dtype)
    
    pAp = np.array([t_Ap, 1.0 - t_Ap], dtype=uv.dtype)
    pBp = np.array([0.0, t_Bp], dtype=uv.dtype)
    pCp = np.array([t_Cp, 0.0], dtype=uv.dtype)
    
    vP = uv_flat - pO
    vAp = pAp - pO
    vBp = pBp - pO
    vCp = pCp - pO
    
    def get_angle(v):
        return np.arctan2(v[..., 1], v[..., 0])
        
    angP = get_angle(vP)
    angAp = get_angle(vAp)
    angBp = get_angle(vBp)
    angCp = get_angle(vCp)
    
    def norm_angle(a, ref):
        res = (a - ref) % (2 * np.pi)
        return np.where(res < 0, res + 2 * np.pi, res)

    pRel = norm_angle(angP, angAp)
    bpRel = norm_angle(angBp, angAp)
    cpRel = norm_angle(angCp, angAp)
    
    labels = np.zeros(uv_flat.shape[0], dtype=np.int32)
    
    maskC = (pRel >= 0) & (pRel < bpRel)
    maskA = (pRel >= bpRel) & (pRel < cpRel)
    maskB = ~(maskC | maskA)
    
    labels[maskA] = 1
    labels[maskB] = 2
    labels[maskC] = 3
    
    labels = labels.reshape(*curr_shape[:-1])
    return labels


def get_triangle_uv(P, A, B, C):
    """
    Computes UV coordinates by projecting to the infinite plane and then
    algebraically clamping within the [0, 1] barycentric triangle bounds.
    """
    uv = get_uv(P, A, B, C)
    is_image = (uv.ndim == 3 and uv.shape[0] == 2 and uv.shape[1] != uv.shape[0])
    
    if is_image:
        u_c = uv[0, ...]
        v_c = uv[1, ...]
    else:
        u_c = uv[..., 0]
        v_c = uv[..., 1]
    
    # Clip algebraically within barycentric boundaries.
    u_c = np.clip(u_c, a_min=0, a_max=None)
    v_c = np.clip(v_c, a_min=0, a_max=None)
    
    exceed = (u_c + v_c) > 1.0
    if exceed.any():
        # Soft map to the hypotenuse
        diff = (u_c[exceed] + v_c[exceed] - 1.0) / 2.0
        u_c[exceed] -= diff
        v_c[exceed] -= diff
        
    if is_image:
        uv_clipped = np.stack([u_c, v_c], axis=0)
    else:
        uv_clipped = np.stack([u_c, v_c], axis=-1)
    return uv_clipped

def project_to_triangle(P, A, B, C):
    """
    Project points P to triangles (A, B, C) using algebraic UV bounds mapping.
    """
    uv = get_triangle_uv(P, A, B, C)
    res = uv_to_3d(uv, A, B, C)
    return np.clip(res, 0, 255).astype(P.dtype)