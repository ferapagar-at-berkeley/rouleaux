from image import Image
from processor import PlaneMask
import torch

import scipy.ndimage as ndimage
import skimage.morphology as morph
import skimage.measure as measure
import skimage.segmentation as seg
import skimage.feature as feature
import skimage.transform as transform
import matplotlib.cm as cm
import numpy as np

class RBCellMask(PlaneMask):
    def __init__(self, name="rbcell", label=1, fill_holes=True, opening_radius=5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.fmt = 'bw'
        self.label = label
        self.fill_holes = fill_holes
        self.opening_radius = opening_radius

    def _process(self, image: Image) -> torch.Tensor:
        cats = super()._process(image)
        mask = (cats == self.label).float()
        
        # Convert to numpy for processing
        mask_np = mask.cpu().numpy()
        
        for c in range(mask_np.shape[0]):
            # 1. Fill internal holes
            if self.fill_holes:
                mask_np[c] = ndimage.binary_fill_holes(mask_np[c])
            
            # 2. Morphological opening to remove small outliers and smooth boundaries
            if self.opening_radius > 0:
                footprint = morph.disk(self.opening_radius)
                mask_np[c] = morph.opening(mask_np[c].astype(bool), footprint=footprint)
            
        mask = torch.from_numpy(mask_np).to(mask.device).float()
            
        return mask

class RBClusters(RBCellMask):
    def __init__(self, name="rbclusters", erosion_radius=7, **kwargs):
        super().__init__(name=name, **kwargs)
        self.erosion_radius = erosion_radius
        self.fmt = "categorical"

    def _process(self, image: Image) -> torch.Tensor:
        # Get binary mask from RBCellMask
        mask = super()._process(image)
        mask_np = mask.cpu().numpy()
        
        labeled_full = np.zeros_like(mask_np, dtype=np.int32)
        total_clusters = 0
        
        footprint = morph.disk(self.erosion_radius) 
        
        for c in range(mask_np.shape[0]):
            chan_mask = mask_np[c] > 0
            
            # 1. Erode to break thin connections
            eroded = morph.erosion(chan_mask, footprint=footprint)
            
            # 2. Label connected components
            labeled, num = measure.label(eroded, return_num=True)
            
            # 3. Dilate labeled components back to their original boundaries
            labeled = seg.expand_labels(labeled, distance=self.erosion_radius)
            # Constrain labels back to the original mask boundaries
            labeled = np.where(chan_mask, labeled, 0)
            
            # Offset labels if there are multiple channels
            labeled_full[c] = np.where(labeled > 0, labeled + total_clusters, 0)
            total_clusters += num
            
        return torch.from_numpy(labeled_full).to(image.data.device)


class ClusterCountMask(RBClusters):
    def __init__(self, name="cluster_counts", max_size=6, min_radius=14, max_radius=22, **kwargs):
        super().__init__(name=name, **kwargs)
        self.max_size = max_size
        self.min_radius = min_radius
        self.max_radius = max_radius
        
        # Create a linear palette for 1 to max_size using the 'plasma' colormap
        cmap = cm.get_cmap("plasma")
        self.palette = {}
        self.legend_labels = {}
        for i in range(1, self.max_size + 1):
            val = (i - 1) / max(1, self.max_size - 1)
            rgba = cmap(val)
            self.palette[i] = [int(c * 255) for c in rgba[:3]]
            
            if i == 1:
                self.legend_labels[i] = "single cell"
            elif i == self.max_size:
                self.legend_labels[i] = f"{i}+ cluster"
            else:
                self.legend_labels[i] = f"{i}-cluster"

    def _process(self, image: Image) -> torch.Tensor:
        # Get individually labeled clusters from RBClusters
        clusters_tensor = super()._process(image)
        clusters_np = clusters_tensor.cpu().numpy()
        
        mask_np = (clusters_np > 0)
        result_np = np.zeros_like(clusters_np, dtype=np.int32)
        
        for c in range(mask_np.shape[0]):
            chan_mask = mask_np[c]
            chan_clusters = clusters_np[c]
            
            if not chan_mask.any():
                continue
                
            # 1. Morphological smoothing to remove noise from boundaries
            smoothed_mask = morph.opening(chan_mask, morph.disk(2))
            edges = feature.canny(smoothed_mask.astype(float))
            
            # 2. Hough Circle Transform (Once per channel)
            hough_radii = np.arange(self.min_radius, self.max_radius, 2)
            hough_res = transform.hough_circle(edges, hough_radii)
            hough_max = hough_res.max()
            
            # 3. Optimized Peak Detection (Once per channel at max sensitivity)
            all_accums, all_cx, all_cy, _ = transform.hough_circle_peaks(
                hough_res, hough_radii, 
                min_xdistance=self.min_radius, 
                min_ydistance=self.min_radius,
                threshold=0.35 * hough_max
            )
            
            # 4. Vectorized Area Calculation
            areas = np.bincount(chan_clusters.ravel())
            
            # 5. Fast Cluster Processing
            unique_clusters = np.unique(chan_clusters)
            avg_r = (self.min_radius + self.max_radius) / 2
            ideal_area = np.pi * (avg_r ** 2)

            # Pre-filter peaks and group them by cluster ID for O(1) counting
            # peaks_by_cluster[cid] = list of (accumulator_value)
            peaks_by_cluster = {}
            for a, x, y in zip(all_accums, all_cx, all_cy):
                if 0 <= y < chan_clusters.shape[0] and 0 <= x < chan_clusters.shape[1]:
                    cid = chan_clusters[y, x]
                    if cid > 0:
                        if cid not in peaks_by_cluster:
                            peaks_by_cluster[cid] = []
                        peaks_by_cluster[cid].append(a)

            # Calculate labels using a lookup table for speed
            lookup = np.zeros(len(areas), dtype=np.int32)
            for cluster_id in unique_clusters:
                if cluster_id == 0: continue
                
                area = areas[cluster_id]
                
                # Ignore tiny clusters that are likely noise ( < 1/3 of a cell)
                if area < (ideal_area / 3):
                    lookup[cluster_id] = 0
                    continue
                
                expected_n = max(1, round(area / ideal_area))
                thresh = 0.55 if expected_n == 1 else (0.35 if expected_n >= 3 else 0.4)
                
                # Count peaks belonging to this cluster that pass the dynamic threshold
                cluster_peaks = peaks_by_cluster.get(cluster_id, [])
                num_cells = sum(1 for a in cluster_peaks if a >= thresh * hough_max)
                
                # Apply safety logic (at least 1 cell, capped at max_size)
                num_cells = min(max(num_cells, 1), self.max_size)
                lookup[cluster_id] = num_cells
                
            # 6. Final assignment (Vectorized lookup)
            result_np[c] = lookup[chan_clusters]
                
        return torch.from_numpy(result_np).to(clusters_tensor.device)

def extract_clusters(image: Image, triangle_path = 'configs/triangle.json', padding: int = 20, fading: float = .7, fade_radius: int = 20) -> tuple[list[Image], dict]:
    """
    Identifies clusters in the image and returns a list of rectangle cutouts 
    for each cluster that doesn't touch the image border, plus a labels dictionary.
    
    Args:
        image: The input Image object.
        triangle_path: Path to the configuration JSON.
        padding: Pixels to add around the cluster bounding box.
        fading: Intensity of black-fading for non-cluster pixels in the cutout (0 to 1).
        
    Returns:
        A tuple (list of Image objects, labels dictionary).
    """
    # 1. Run processors if not already present
    if 'rbclusters' not in image.versions:
        RBClusters.from_json(triangle_path).process(image)
    labeled_np = image.versions['rbclusters'].data.cpu().numpy()
    
    if 'cluster_counts' not in image.versions:
        ClusterCountMask.from_json(triangle_path).process(image)
    counts_np = image.versions['cluster_counts'].data.cpu().numpy()
    
    # original image data (usually [C, H, W])
    orig_data = image.data.clone().float()
    C, H, W = orig_data.shape
    orig_np = orig_data.cpu().numpy()
    
    results = []
    labels = {}
    
    # labeled_np has shape [C, H, W]
    for c in range(labeled_np.shape[0]):
        chan_labels = labeled_np[c]
        chan_counts = counts_np[c]
        
        if not (chan_labels > 0).any():
            continue
            
        props = measure.regionprops(chan_labels)
        
        for prop in props:
            label_id = prop.label
            min_r, min_c, max_r, max_c = prop.bbox
            
            # Check if cluster itself touches border
            if min_r == 0 or min_c == 0 or max_r == H or max_c == W:
                continue
                
            # Get predicted size for this cluster from ClusterCountMask
            # All pixels in the cluster share the same predicted count
            coords = prop.coords[0] # Take first pixel coordinate
            predicted_size = int(chan_counts[coords[0], coords[1]])
            
            # Determine padded coordinates
            p_min_r = max(0, min_r - padding)
            p_min_c = max(0, min_c - padding)
            p_max_r = min(H, max_r + padding)
            p_max_c = min(W, max_c + padding)
            
            # Extract cutout from original data
            cutout = orig_np[:, p_min_r:p_max_r, p_min_c:p_max_c].copy()
            
            # 1. Extract binary mask for this cluster and smooth it with a 10x10 box blur
            mask_binary = (chan_labels[p_min_r:p_max_r, p_min_c:p_max_c] == label_id).astype(float)
            soft_mask = ndimage.uniform_filter(mask_binary, size=fade_radius)
            
            # 2. Interpolate: result = color * [soft_mask + fading * (1 - soft_mask)]
            weight = soft_mask + fading * (1.0 - soft_mask)
            cutout = cutout * weight # Broadcasts across channels
            
            # Create new Image object
            new_title = f"{image.title}_cluster_{label_id}"
            new_img = Image(
                torch.from_numpy(cutout).to(image.data.device),
                title=new_title,
                fmt=image.format,
                palette=image.palette,
                legend_labels=image.legend_labels
            )
            
            results.append(new_img)
            labels[new_title] = [predicted_size]
            
    return results, labels
