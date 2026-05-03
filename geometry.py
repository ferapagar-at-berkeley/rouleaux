import torch

def get_uv(P, A, B, C):
    """
    Returns (u,v) where P_proj = A + u*(B-A) + v*(C-A)
    Supports: [3], [N, 3], and [3, H, W]
    """
    is_image = (P.dim() == 3 and P.shape[0] == 3 and P.shape[1] != P.shape[0])
    if is_image:
        P = P.permute(1, 2, 0)
    curr_shape = P.shape
    P_flat = P.reshape(-1, 3) if P.dim() > 1 else P.unsqueeze(0)
    
    A, B, C = A.view(1, 3).float(), B.view(1, 3).float(), C.view(1, 3).float()
    
    v0 = B - A
    v1 = C - A
    v2 = P_flat - A
    
    d00 = torch.sum(v0 * v0, dim=-1)
    d01 = torch.sum(v0 * v1, dim=-1)
    d11 = torch.sum(v1 * v1, dim=-1)
    d20 = torch.sum(v2 * v0, dim=-1)
    d21 = torch.sum(v2 * v1, dim=-1)
    
    denom = d00 * d11 - d01 * d01 + 1e-10
    
    u = (d11 * d20 - d01 * d21) / denom
    v = (d00 * d21 - d01 * d20) / denom
    
    uv = torch.stack([u, v], dim=-1)
    
    uv = uv.view(*curr_shape[:-1], 2)
    if is_image:
        uv = uv.permute(2, 0, 1) # [2, H, W]
    
    return uv

def uv_to_3d(uv, A, B, C):
    """
    Converts (u, v) coordinates back to 3D space: P = A + u*(B-A) + v*(C-A)
    uv supports shape [..., 2] or [2, H, W]
    """
    is_image = (uv.dim() == 3 and uv.shape[0] == 2 and uv.shape[1] != uv.shape[0])
    
    A, B, C = A.view(-1, 3).float(), B.view(-1, 3).float(), C.view(-1, 3).float()
    
    if is_image:
        u = uv[0:1, ...]
        v = uv[1:2, ...]
        A_val = A.view(3, 1, 1)
        B_val = B.view(3, 1, 1)
        C_val = C.view(3, 1, 1)
    else:
        u = uv[..., 0:1]
        v = uv[..., 1:2]
        A_val = A.view(1, 3) # Or broadcast natively
        B_val = B.view(1, 3)
        C_val = C.view(1, 3)
        
    return A_val + u * (B_val - A_val) + v * (C_val - A_val)

def project_to_plane(P, A, B, C):
    """
    Projects points P onto the infinite plane defined by A, B, C using UV space.
    """
    uv = get_uv(P, A, B, C)
    res = uv_to_3d(uv, A, B, C)
    return torch.clamp(res, 0, 255).to(P.dtype)

def mask_plane_labels(uv, v_O, t_Ap, t_Bp, t_Cp):
    """
    Returns tensor of same spatial shape as uv, with values 1, 2, or 3 based on angular sectors.
    Used for both Plane and Triangle classification modes.
    Assumes uv is already generated via get_uv() or project_to_triangle().
    """
    is_image = (uv.dim() == 3 and uv.shape[0] == 2 and uv.shape[1] != uv.shape[0])
    
    if is_image:
        uv = uv.permute(1, 2, 0)
    curr_shape = uv.shape
    uv_flat = uv.reshape(-1, 2)
    
    u_o = v_O[1]
    v_o = v_O[2]
    pO = torch.tensor([u_o, v_o], device=uv.device, dtype=uv.dtype)
    
    pAp = torch.tensor([t_Ap, 1.0 - t_Ap], device=uv.device, dtype=uv.dtype)
    pBp = torch.tensor([0.0, t_Bp], device=uv.device, dtype=uv.dtype)
    pCp = torch.tensor([t_Cp, 0.0], device=uv.device, dtype=uv.dtype)
    
    vP = uv_flat - pO
    vAp = pAp - pO
    vBp = pBp - pO
    vCp = pCp - pO
    
    def get_angle(v):
        return torch.atan2(v[..., 1], v[..., 0])
        
    angP = get_angle(vP)
    angAp = get_angle(vAp)
    angBp = get_angle(vBp)
    angCp = get_angle(vCp)
    
    def norm_angle(a, ref):
        res = (a - ref) % (2 * torch.pi)
        res[res < 0] += 2 * torch.pi
        return res

    pRel = norm_angle(angP, angAp)
    bpRel = norm_angle(angBp, angAp)
    cpRel = norm_angle(angCp, angAp)
    
    labels = torch.zeros(uv_flat.shape[0], dtype=torch.int32, device=uv.device)
    
    maskC = (pRel >= 0) & (pRel < bpRel)
    maskA = (pRel >= bpRel) & (pRel < cpRel)
    maskB = ~(maskC | maskA)
    
    labels[maskA] = 1
    labels[maskB] = 2
    labels[maskC] = 3
    
    labels = labels.view(*curr_shape[:-1])
    return labels


def get_triangle_uv(P, A, B, C):
    """
    Computes UV coordinates by projecting to the infinite plane and then
    algebraically clamping within the [0, 1] barycentric triangle bounds.
    """
    uv = get_uv(P, A, B, C)
    is_image = (uv.dim() == 3 and uv.shape[0] == 2 and uv.shape[1] != uv.shape[0])
    
    if is_image:
        u_c = uv[0, ...]
        v_c = uv[1, ...]
    else:
        u_c = uv[..., 0]
        v_c = uv[..., 1]
    
    # Clip algebraically within barycentric boundaries.
    u_c = torch.clamp(u_c, min=0)
    v_c = torch.clamp(v_c, min=0)
    
    exceed = (u_c + v_c) > 1.0
    if exceed.any():
        # Soft map to the hypotenuse
        diff = (u_c[exceed] + v_c[exceed] - 1.0) / 2.0
        u_c[exceed] -= diff
        v_c[exceed] -= diff
        
    if is_image:
        uv_clipped = torch.stack([u_c, v_c], dim=0)
    else:
        uv_clipped = torch.stack([u_c, v_c], dim=-1)
    return uv_clipped

def project_to_triangle(P, A, B, C):
    """
    Project points P to triangles (A, B, C) using algebraic UV bounds mapping.
    """
    uv = get_triangle_uv(P, A, B, C)
    res = uv_to_3d(uv, A, B, C)
    return torch.clamp(res, 0, 255).to(P.dtype)