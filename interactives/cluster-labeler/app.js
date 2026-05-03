// State
let allLoadedImages = []; // Full set of File objects
let images = []; // Filtered set of File objects
let currentIndex = -1;
let labels = {}; // key: image filename, value: array of integers
let predictions = {}; // key: image filename, value: array of integers
let sessionKeys = null; // Set of filenames for the current filtered session

// DOM Elements
const imageInput = document.getElementById('imageInput');
const jsonInputLabel = document.getElementById('jsonInputLabel');
const jsonInputPred = document.getElementById('jsonInputPred');
const saveJsonBtn = document.getElementById('saveJsonBtn');
const copyJsonBtnLabel = document.getElementById('copyJsonBtnLabel');
const copyJsonBtnPred = document.getElementById('copyJsonBtnPred');
const prevBtn = document.getElementById('prevBtn');
const nextBtn = document.getElementById('nextBtn');
const imageSelect = document.getElementById('imageSelect');
const clusterImage = document.getElementById('clusterImage');
const noImageText = document.getElementById('noImageText');
const progressIndicator = document.getElementById('progressIndicator');
const labelInput = document.getElementById('labelInput');
const predictionInput = document.getElementById('predictionInput');
const viewFilter = document.getElementById('viewFilter');
const imageGrid = document.getElementById('imageGrid');

let gridUrls = []; // Track URLs to revoke them

// 1. Load Images
imageInput.addEventListener('change', (e) => {
    if (e.target.files.length === 0) return;
    allLoadedImages = Array.from(e.target.files).sort((a, b) => a.name.localeCompare(b.name));
    labelInput.disabled = false;
    sessionKeys = null;
    updateFilteredList();
});

function updateFilteredList(keepCurrent = false, forceRefresh = false) {
    const prevFile = images[currentIndex];
    const filter = viewFilter.value;

    if (filter === 'all') {
        images = [...allLoadedImages];
        sessionKeys = null;
    } else {
        // If we just switched to this filter or forced a refresh (e.g. on JSON load)
        if (!sessionKeys || forceRefresh) {
            sessionKeys = new Set();
            allLoadedImages.forEach(file => {
                const nameKey = file.name.replace(/\.[^/.]+$/, "");
                if (filter === 'unlabeled') {
                    if (!labels[nameKey]) sessionKeys.add(file.name);
                } else if (filter === 'incorrects') {
                    const l = labels[nameKey] ? labels[nameKey].join(',') : '';
                    const p = predictions[nameKey] ? predictions[nameKey].join(',') : '';
                    if (l !== p) sessionKeys.add(file.name);
                }
            });
        }
        images = allLoadedImages.filter(file => sessionKeys.has(file.name));
    }

    imageSelect.innerHTML = '';
    images.forEach((file, index) => {
        const option = document.createElement('option');
        option.value = index;
        option.textContent = file.name;
        imageSelect.appendChild(option);
    });

    if (images.length === 0) {
        currentIndex = -1;
        clusterImage.style.display = 'none';
        noImageText.style.display = 'block';
        noImageText.textContent = "No images match the filter";
        progressIndicator.textContent = "Image 0 of 0: None";
        labelInput.value = '';
        predictionInput.value = 'N/A';
        return;
    }

    if (keepCurrent && prevFile) {
        const newIdx = images.findIndex(f => f.name === prevFile.name);
        if (newIdx >= 0) {
            goToImage(newIdx);
            return;
        }
    }
    goToImage(0);
}

viewFilter.addEventListener('change', () => {
    sessionKeys = null; // Reset session set when filter changes
    updateFilteredList();
});

// 2. Load JSON
function handleJsonLoad(e, targetState) {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
        try {
            const data = JSON.parse(event.target.result);
            if (targetState === 'labels') {
                labels = { ...labels, ...data };
            } else {
                predictions = { ...predictions, ...data };
            }
            // alert('JSON loaded successfully!');
            updateFilteredList(true, true); // Re-filter and force refresh the session set
        } catch (err) {
            alert('Invalid JSON file.');
        }
    };
    reader.readAsText(file);
}

jsonInputLabel.addEventListener('change', (e) => handleJsonLoad(e, 'labels'));
jsonInputPred.addEventListener('change', (e) => handleJsonLoad(e, 'predictions'));

// 3. Save JSON (Ground Truth)
saveJsonBtn.addEventListener('click', () => {
    const dataStr = JSON.stringify(labels, null, 2);
    const blob = new Blob([dataStr], { type: "application/json" });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = 'labels.json';
    a.click();

    URL.revokeObjectURL(url);
});

// 4. Copy JSON
async function handleCopy(targetState, btn) {
    const data = targetState === 'labels' ? labels : predictions;
    const dataStr = JSON.stringify(data, null, 2);
    try {
        await navigator.clipboard.writeText(dataStr);
        const originalText = btn.textContent;
        btn.textContent = 'Copied!';
        setTimeout(() => btn.textContent = originalText, 2000);
    } catch (err) {
        alert('Failed to copy to clipboard.');
    }
}

copyJsonBtnLabel.addEventListener('click', () => handleCopy('labels', copyJsonBtnLabel));
copyJsonBtnPred.addEventListener('click', () => handleCopy('predictions', copyJsonBtnPred));

// Navigation Functions
function goToImage(index) {
    if (images.length === 0) return;

    // Wrap around logic
    if (index < 0) {
        index = images.length - 1;
    } else if (index >= images.length) {
        index = 0;
    }

    currentIndex = index;
    const file = images[currentIndex];
    const fileName = file.name;
    const nameKey = fileName.replace(/\.[^/.]+$/, ""); // Strip extension

    // Render Image
    const objectUrl = URL.createObjectURL(file);
    clusterImage.src = objectUrl;
    clusterImage.style.display = 'block';
    noImageText.style.display = 'none';

    // Memory cleanup for previous object URL
    clusterImage.onload = () => URL.revokeObjectURL(objectUrl);

    // Update UI
    progressIndicator.textContent = `Image ${currentIndex + 1} of ${images.length}: ${nameKey}`;
    imageSelect.value = currentIndex;

    // Retrieve Label and Prediction
    if (labels[nameKey]) {
        labelInput.value = labels[nameKey].join(', ');
    } else {
        labelInput.value = '';
    }

    if (predictions[nameKey]) {
        predictionInput.value = predictions[nameKey].join(', ');
    } else {
        predictionInput.value = 'N/A';
    }

    // Auto-focus and auto-select text
    labelInput.focus();
    labelInput.select();

    // Render Grid
    renderGrid();
}

function renderGrid() {
    if (!imageGrid) return;

    // Cleanup previous grid URLs
    gridUrls.forEach(url => URL.revokeObjectURL(url));
    gridUrls = [];
    imageGrid.innerHTML = '';

    if (images.length === 0) return;

    let startIndex;
    if (images.length <= 36) {
        startIndex = 0;
    } else {
        startIndex = currentIndex - 4;
    }

    const totalItems = 36; // 4 rows * 9 cols

    for (let i = 0; i < totalItems; i++) {
        const targetIndex = startIndex + i;
        const gridItem = document.createElement('div');
        gridItem.className = 'grid-item';

        if (targetIndex >= 0 && targetIndex < images.length) {
            const file = images[targetIndex];
            const nameKey = file.name.replace(/\.[^/.]+$/, "");

            if (targetIndex === currentIndex) {
                gridItem.classList.add('current');
            }

            const img = document.createElement('img');
            const url = URL.createObjectURL(file);
            gridUrls.push(url);
            img.src = url;

            const lVal = labels[nameKey] ? labels[nameKey].join(',') : '-';
            const pVal = predictions[nameKey] ? predictions[nameKey].join(',') : '-';

            const subtitle = document.createElement('div');
            subtitle.className = 'grid-subtitle';
            subtitle.textContent = `Label:${lVal} Pred:${pVal}`;

            const title = document.createElement('div');
            title.className = 'grid-title';
            title.textContent = nameKey;

            gridItem.appendChild(img);
            gridItem.appendChild(title);
            gridItem.appendChild(subtitle);
            gridItem.onclick = () => saveAndGo(targetIndex);
        } else {
            gridItem.style.visibility = 'hidden';
        }
        imageGrid.appendChild(gridItem);
    }
}

prevBtn.addEventListener('click', () => saveAndGo(currentIndex - 1));
nextBtn.addEventListener('click', () => saveAndGo(currentIndex + 1));
imageSelect.addEventListener('change', (e) => saveAndGo(parseInt(e.target.value)));

// Save current input and navigate
function saveAndGo(nextIndex) {
    if (currentIndex >= 0) {
        const file = images[currentIndex];
        const nameKey = file.name.replace(/\.[^/.]+$/, "");

        const rawInput = labelInput.value.trim();
        if (rawInput) {
            // Parse into array of numbers, filter out non-numbers
            const numbers = rawInput.split(/[\s,]+/).filter(s => s !== '').map(Number).filter(n => !isNaN(n));
            labels[nameKey] = numbers;
        }
    }

    // Move to the next image in the stable filtered list
    goToImage(nextIndex);
}

// Rapid Labeling (Enter / Shift+Enter)
labelInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        e.preventDefault();
        if (e.shiftKey) {
            saveAndGo(currentIndex - 1);
        } else {
            saveAndGo(currentIndex + 1);
        }
    }
});

// Handle global arrow keys when not typing
document.addEventListener('keydown', (e) => {
    if (document.activeElement !== labelInput) {
        if (e.key === 'ArrowRight') {
            saveAndGo(currentIndex + 1);
        } else if (e.key === 'ArrowLeft') {
            saveAndGo(currentIndex - 1);
        }
    }
});
