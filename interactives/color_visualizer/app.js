const state = {
    colorA: hexToRgb(document.getElementById('colorA').value),
    colorB: hexToRgb(document.getElementById('colorB').value),
    colorC: hexToRgb(document.getElementById('colorC').value),
    imgDataOriginal: null,
    imageW: 0,
    imageH: 0
};

const tCanvas = document.getElementById('triangleCanvas');
const tCtx = tCanvas.getContext('2d');
const cw = tCanvas.width;
const ch = tCanvas.height;

// Equilateral-ish triangle bounds mapped directly to A, B, C
const vA = { x: cw / 2, y: 40 };
const vB = { x: cw - 40, y: ch - 40 };
const vC = { x: 40, y: ch - 40 };

// Draggable points. 't' indicates parameter along the edge [0, 1]
const handles = {
    O: { 
        x: 0.2301 * vA.x + 0.6454 * vB.x + 0.1245 * vC.x, 
        y: 0.2301 * vA.y + 0.6454 * vB.y + 0.1245 * vC.y, 
        type: 'free' 
    },
    Ap: { t: 0.7794 }, // On BC (C -> B)
    Bp: { t: 0.2617 }, // On AC (A -> C)
    Cp: { t: 0.5023 }, // On AB (A -> B)
};

function getHandlePositions() {
    return {
        O: handles.O,
        Ap: {
            x: vC.x + (vB.x - vC.x) * handles.Ap.t,
            y: vC.y + (vB.y - vC.y) * handles.Ap.t
        },
        Bp: {
            x: vA.x + (vC.x - vA.x) * handles.Bp.t,
            y: vA.y + (vC.y - vA.y) * handles.Bp.t
        },
        Cp: {
            x: vA.x + (vB.x - vA.x) * handles.Cp.t,
            y: vA.y + (vB.y - vA.y) * handles.Cp.t
        }
    };
}

function drawTriangle() {
    let imgData = tCtx.createImageData(cw, ch);
    let pd = imgData.data;

    // Fill the Barycentric Canvas manually per pixel for accuracy
    for (let y = 0; y < ch; y++) {
        for (let x = 0; x < cw; x++) {
            let idx = (y * cw + x) * 4;
            let bary = getBarycentric2D([x, y], [vA.x, vA.y], [vB.x, vB.y], [vC.x, vC.y]);

            let r = bary[0] * state.colorA[0] + bary[1] * state.colorB[0] + bary[2] * state.colorC[0];
            let g = bary[0] * state.colorA[1] + bary[1] * state.colorB[1] + bary[2] * state.colorC[1];
            let b = bary[0] * state.colorA[2] + bary[1] * state.colorB[2] + bary[2] * state.colorC[2];

            // Show all valid RGB colors even if they fall outside the 0-1 barycentric bounds
            if (r >= -0.5 && r <= 255.5 && g >= -0.5 && g <= 255.5 && b >= -0.5 && b <= 255.5) {
                pd[idx] = clamp(r, 0, 255);
                pd[idx + 1] = clamp(g, 0, 255);
                pd[idx + 2] = clamp(b, 0, 255);

                // Add transparency to colors outside the triangle
                if (bary[0] >= -0.005 && bary[1] >= -0.005 && bary[2] >= -0.005) {
                    pd[idx + 3] = 255;
                } else {
                    pd[idx + 3] = 120; // Semi-transparent
                }
            } else {
                pd[idx + 3] = 0;
            }
        }
    }
    tCtx.putImageData(imgData, 0, 0);

    // Draw Triangle Perimeter
    tCtx.shadowColor = "rgba(0,0,0,0.6)";
    tCtx.shadowBlur = 4;
    tCtx.lineWidth = 2;
    tCtx.strokeStyle = 'rgba(255, 255, 255, 1.0)';
    tCtx.setLineDash([5, 4]);
    tCtx.beginPath();
    tCtx.moveTo(vA.x, vA.y);
    tCtx.lineTo(vB.x, vB.y);
    tCtx.lineTo(vC.x, vC.y);
    tCtx.closePath();
    tCtx.stroke();
    tCtx.setLineDash([]);
    tCtx.shadowBlur = 0;

    let pos = getHandlePositions();

    // Draw Mask boundary segments (rays extended to screen edge)
    tCtx.lineWidth = 2;
    tCtx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
    tCtx.beginPath();

    let extAp = { x: pos.O.x + (pos.Ap.x - pos.O.x) * 50, y: pos.O.y + (pos.Ap.y - pos.O.y) * 50 };
    let extBp = { x: pos.O.x + (pos.Bp.x - pos.O.x) * 50, y: pos.O.y + (pos.Bp.y - pos.O.y) * 50 };
    let extCp = { x: pos.O.x + (pos.Cp.x - pos.O.x) * 50, y: pos.O.y + (pos.Cp.y - pos.O.y) * 50 };

    tCtx.moveTo(pos.O.x, pos.O.y);
    tCtx.lineTo(extAp.x, extAp.y);
    tCtx.moveTo(pos.O.x, pos.O.y);
    tCtx.lineTo(extBp.x, extBp.y);
    tCtx.moveTo(pos.O.x, pos.O.y);
    tCtx.lineTo(extCp.x, extCp.y);
    tCtx.stroke();

    // Draw Handles
    let points = [pos.O, pos.Ap, pos.Bp, pos.Cp];
    points.forEach((p, i) => {
        tCtx.beginPath();
        tCtx.arc(p.x, p.y, i === 0 ? 8 : 6, 0, Math.PI * 2);
        tCtx.fillStyle = '#38bdf8';
        tCtx.fill();
        tCtx.lineWidth = 2;
        tCtx.strokeStyle = 'white';
        tCtx.stroke();
    });

    // Vertex Labels
    tCtx.font = "bold 18px Inter";
    tCtx.fillStyle = 'white';
    tCtx.shadowColor = "rgba(0,0,0,0.5)";
    tCtx.shadowBlur = 4;
    tCtx.fillText("A", vA.x - 6, vA.y - 15);
    tCtx.fillText("B", vB.x + 15, vB.y + 10);
    tCtx.fillText("C", vC.x - 25, vC.y + 10);
    tCtx.shadowBlur = 0;

    // Update Parameter Info
    document.getElementById('valAp').innerText = handles.Ap.t.toFixed(2);
    document.getElementById('valBp').innerText = handles.Bp.t.toFixed(2);
    document.getElementById('valCp').innerText = handles.Cp.t.toFixed(2);

    let baryO = getBarycentric2D([handles.O.x, handles.O.y], [vA.x, vA.y], [vB.x, vB.y], [vC.x, vC.y]);
    document.getElementById('valVo').innerText = `${baryO[0].toFixed(2)}, ${baryO[1].toFixed(2)}, ${baryO[2].toFixed(2)}`;
}

// Image Subsystem
const imgCanvasOriginal = document.getElementById('imageCanvasOriginal');
const iCtxOriginal = imgCanvasOriginal.getContext('2d');
const imgCanvasProcessed = document.getElementById('imageCanvasProcessed');
const iCtxProcessed = imgCanvasProcessed.getContext('2d');

document.getElementById('imageUpload').addEventListener('change', e => {
    let file = e.target.files[0];
    if (!file) return;
    let img = new Image();
    img.onload = () => {
        state.imageW = img.width;
        state.imageH = img.height;

        imgCanvasOriginal.width = img.width;
        imgCanvasOriginal.height = img.height;
        imgCanvasProcessed.width = img.width;
        imgCanvasProcessed.height = img.height;

        iCtxOriginal.drawImage(img, 0, 0);
        state.imgDataOriginal = iCtxOriginal.getImageData(0, 0, img.width, img.height);
        processImage(); // Process on initial upload
    };
    img.src = URL.createObjectURL(file);
});

function rgbToHex(rgb) {
    return "#" + rgb.map(x => {
        const hex = x.toString(16);
        return hex.length === 1 ? "0" + hex : hex;
    }).join("");
}

function processImage() {
    if (!state.imgDataOriginal) return;

    // Use setTimeout so the UI can paint any updates before freezing the thread
    setTimeout(() => {
        let newData = new ImageData(
            new Uint8ClampedArray(state.imgDataOriginal.data),
            state.imageW, state.imageH
        );
        let d = newData.data;
        let mode = document.getElementById('viewMode').value;

        if (mode === 'original') {
            iCtxProcessed.putImageData(state.imgDataOriginal, 0, 0);
        } else {
            let baryO = getBarycentric2D([handles.O.x, handles.O.y], [vA.x, vA.y], [vB.x, vB.y], [vC.x, vC.y]);

            for (let i = 0; i < d.length; i += 4) {
                let p = [d[i], d[i + 1], d[i + 2]];
                let res = p;

                if (mode === 'plane') {
                    res = projectToPlane(p, state.colorA, state.colorB, state.colorC);
                } else if (mode === 'triangle') {
                    res = projectToTriangle(p, state.colorA, state.colorB, state.colorC);
                } else if (mode === 'mask') {
                    res = maskTriangle(
                        p, state.colorA, state.colorB, state.colorC,
                        baryO, handles.Ap.t, handles.Bp.t, handles.Cp.t
                    );
                } else if (mode === 'mask-plane') {
                    res = maskPlane(
                        p, state.colorA, state.colorB, state.colorC,
                        baryO, handles.Ap.t, handles.Bp.t, handles.Cp.t
                    );
                }

                d[i] = clamp(res[0], 0, 255);
                d[i + 1] = clamp(res[1], 0, 255);
                d[i + 2] = clamp(res[2], 0, 255);
            }
            iCtxProcessed.putImageData(newData, 0, 0);
        }
    }, 10);
}

document.getElementById('viewMode').addEventListener('change', processImage);

document.getElementById('copyJsonBtn').addEventListener('click', () => {
    let baryO = getBarycentric2D([handles.O.x, handles.O.y], [vA.x, vA.y], [vB.x, vB.y], [vC.x, vC.y]);

    let config = {
        A: state.colorA,
        B: state.colorB,
        C: state.colorC,
        t_Ap: parseFloat(handles.Ap.t.toFixed(4)),
        t_Bp: parseFloat(handles.Bp.t.toFixed(4)),
        t_Cp: parseFloat(handles.Cp.t.toFixed(4)),
        v_O: baryO.map(v => parseFloat(v.toFixed(4)))
    };

    let json = JSON.stringify(config, null, 4);
    navigator.clipboard.writeText(json).then(() => {
        const btn = document.getElementById('copyJsonBtn');
        const oldText = btn.innerText;
        btn.innerText = "Copied!";
        btn.style.borderColor = "#22c55e"; // Success green
        setTimeout(() => {
            btn.innerText = oldText;
            btn.style.borderColor = "var(--glass-border)";
        }, 2000);
    });
});

document.getElementById('loadJsonBtn').addEventListener('click', () => {
    const input = document.getElementById('jsonInput').value;
    try {
        const config = JSON.parse(input);

        // Update Colors (checking new and old names for safety)
        const A = config.A || config.colorA;
        if (A) {
            state.colorA = A;
            document.getElementById('colorA').value = rgbToHex(state.colorA);
        }
        const B = config.B || config.colorB;
        if (B) {
            state.colorB = B;
            document.getElementById('colorB').value = rgbToHex(state.colorB);
        }
        const C = config.C || config.colorC;
        if (C) {
            state.colorC = C;
            document.getElementById('colorC').value = rgbToHex(state.colorC);
        }

        // Update T's
        const tAp = config.t_Ap !== undefined ? config.t_Ap : config.tA_prime;
        if (tAp !== undefined) handles.Ap.t = tAp;
        const tBp = config.t_Bp !== undefined ? config.t_Bp : config.tB_prime;
        if (tBp !== undefined) handles.Bp.t = tBp;
        const tCp = config.t_Cp !== undefined ? config.t_Cp : config.tC_prime;
        if (tCp !== undefined) handles.Cp.t = tCp;

        // Update O position from v_O (barycentric)
        const vO = config.v_O || config.v_o;
        if (vO) {
            const [u, v, w] = vO;
            handles.O.x = u * vA.x + v * vB.x + w * vC.x;
            handles.O.y = u * vA.y + v * vB.y + w * vC.y;
        }

        drawTriangle();
        processImage();

        const btn = document.getElementById('loadJsonBtn');
        btn.innerText = "Loaded!";
        setTimeout(() => btn.innerText = "Load Config", 2000);
    } catch (e) {
        alert("Invalid JSON format!");
        console.error(e);
    }
});

// Interaction Logic
let dragging = null;

tCanvas.addEventListener('mousedown', e => {
    let r = tCanvas.getBoundingClientRect();
    let x = (e.clientX - r.left) * (tCanvas.width / r.width);
    let y = (e.clientY - r.top) * (tCanvas.height / r.height);

    let pos = getHandlePositions();
    let dists = Object.keys(pos).map(k => ({
        id: k,
        d: Math.hypot(pos[k].x - x, pos[k].y - y)
    }));
    dists.sort((a, b) => a.d - b.d);

    if (dists[0].d < 15) {
        dragging = dists[0].id;
    }
});

tCanvas.addEventListener('mousemove', e => {
    if (!dragging) return;
    let r = tCanvas.getBoundingClientRect();
    let x = (e.clientX - r.left) * (tCanvas.width / r.width);
    let y = (e.clientY - r.top) * (tCanvas.height / r.height);

    if (dragging === 'O') {
        let bary = getBarycentric2D([x, y], [vA.x, vA.y], [vB.x, vB.y], [vC.x, vC.y]);
        if (bary[0] >= 0 && bary[1] >= 0 && bary[2] >= 0) {
            handles.O.x = x;
            handles.O.y = y;
            drawTriangle();
        }
    } else {
        let ptA, ptB;
        if (dragging === 'Ap') { ptA = vC; ptB = vB; }
        else if (dragging === 'Bp') { ptA = vA; ptB = vC; }
        else if (dragging === 'Cp') { ptA = vA; ptB = vB; }

        let ab = sub([ptB.x, ptB.y, 0], [ptA.x, ptA.y, 0]);
        let ap = sub([x, y, 0], [ptA.x, ptA.y, 0]);
        let t = dot(ap, ab) / dot(ab, ab);
        handles[dragging].t = clamp(t, 0.01, 0.99); // Avoid extreme corners
        drawTriangle();
    }
});

window.addEventListener('mouseup', () => {
    if (dragging) {
        dragging = null;
        processImage();
    }
});

// Bind input updates
['A', 'B', 'C'].forEach(l => {
    document.getElementById('color' + l).addEventListener('input', e => {
        state['color' + l] = hexToRgb(e.target.value);
        drawTriangle();
        processImage();
    });
});

// Init
drawTriangle();

