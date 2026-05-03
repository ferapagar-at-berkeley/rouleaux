// Vector Math Utilities
function add(a, b) {
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function sub(a, b) {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function dot(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function scale(v, s) {
    return [v[0] * s, v[1] * s, v[2] * s];
}

function hexToRgb(hex) {
    let bigint = parseInt(hex.substring(1), 16);
    return [(bigint >> 16) & 255, (bigint >> 8) & 255, bigint & 255];
}

function clamp(v, min, max) {
    return Math.max(min, Math.min(max, v));
}

// 2D Barycentric for rendering the UI triangle canvas
function getBarycentric2D(p, a, b, c) {
    let v0 = [b[0] - a[0], b[1] - a[1]];
    let v1 = [c[0] - a[0], c[1] - a[1]];
    let v2 = [p[0] - a[0], p[1] - a[1]];
    
    let d00 = v0[0]*v0[0] + v0[1]*v0[1];
    let d01 = v0[0]*v1[0] + v0[1]*v1[1];
    let d11 = v1[0]*v1[0] + v1[1]*v1[1];
    let d20 = v2[0]*v0[0] + v2[1]*v0[1];
    let d21 = v2[0]*v1[0] + v2[1]*v1[1];
    
    let denom = d00 * d11 - d01 * d01;
    if (Math.abs(denom) < 1e-10) return [1, 0, 0];
    
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let u = 1.0 - v - w;
    return [u, v, w];
}

// Core UV Mapping (Mirroring Python get_uv)
// Returns [u, v] where u is weight of B and v is weight of C
function getUV(P, A, B, C) {
    let v0 = sub(B, A);
    let v1 = sub(C, A);
    let v2 = sub(P, A);
    
    let d00 = dot(v0, v0);
    let d01 = dot(v0, v1);
    let d11 = dot(v1, v1);
    let d20 = dot(v2, v0);
    let d21 = dot(v2, v1);
    
    let denom = d00 * d11 - d01 * d01;
    if (Math.abs(denom) < 1e-10) return [0, 0];
    
    let u = (d11 * d20 - d01 * d21) / denom;
    let v = (d00 * d21 - d01 * d20) / denom;
    return [u, v];
}

// Triangle-Clamped UV Mapping (Mirroring Python get_triangle_uv)
function getTriangleUV(P, A, B, C) {
    let uv = getUV(P, A, B, C);
    let u = Math.max(0, uv[0]);
    let v = Math.max(0, uv[1]);
    
    if (u + v > 1.0) {
        let diff = (u + v - 1.0) / 2.0;
        u -= diff;
        v -= diff;
    }
    return [u, v];
}

// 3D Reconstruction (Mirroring Python uv_to_3d)
function uvTo3D(uv, A, B, C) {
    let u = uv[0];
    let v = uv[1];
    let AB = sub(B, A);
    let AC = sub(C, A);
    return add(A, add(scale(AB, u), scale(AC, v)));
}

// Angular Sector Classification (Mirroring Python mask_plane_labels)
function getPlaneLabel(uv, baryO, t_Ap, t_Bp, t_Cp) {
    // uv is [u, v] (weights of B, C)
    // pO uses bary[1], bary[2] which are weights of B, C
    let pO = [baryO[1], baryO[2]];
    
    // Boundary points in UV space
    let pAp = [t_Ap, 1.0 - t_Ap]; // On BC
    let pBp = [0.0, t_Bp];        // On AC
    let pCp = [t_Cp, 0.0];        // On AB
    
    let vP = [uv[0] - pO[0], uv[1] - pO[1]];
    let vAp = [pAp[0] - pO[0], pAp[1] - pO[1]];
    let vBp = [pBp[0] - pO[0], pBp[1] - pO[1]];
    let vCp = [pCp[0] - pO[0], pCp[1] - pO[1]];
    
    const angle = (v) => Math.atan2(v[1], v[0]);
    const angP = angle(vP);
    const angAp = angle(vAp);
    const angBp = angle(vBp);
    const angCp = angle(vCp);
    
    const normAngle = (a, ref) => {
        let res = (a - ref) % (Math.PI * 2);
        if (res < 0) res += Math.PI * 2;
        return res;
    };
    
    let pRel = normAngle(angP, angAp);
    let bpRel = normAngle(angBp, angAp);
    let cpRel = normAngle(angCp, angAp);
    
    // Sector mapping
    if (pRel >= 0 && pRel < bpRel) return 3; // C
    if (pRel >= bpRel && pRel < cpRel) return 1; // A
    return 2; // B
}

// High-level filter entry points used by app.js

function projectToPlane(P, A, B, C) {
    let uv = getUV(P, A, B, C);
    return uvTo3D(uv, A, B, C);
}

function projectToTriangle(P, A, B, C) {
    let uv = getTriangleUV(P, A, B, C);
    return uvTo3D(uv, A, B, C);
}

function maskPlane(P, A, B, C, baryO, t_Ap, t_Bp, t_Cp) {
    let uv = getUV(P, A, B, C);
    let label = getPlaneLabel(uv, baryO, t_Ap, t_Bp, t_Cp);
    if (label === 1) return A;
    if (label === 2) return B;
    return C;
}

function maskTriangle(P, A, B, C, baryO, t_Ap, t_Bp, t_Cp) {
    let uv = getTriangleUV(P, A, B, C);
    let label = getPlaneLabel(uv, baryO, t_Ap, t_Bp, t_Cp);
    if (label === 1) return A;
    if (label === 2) return B;
    return C;
}
