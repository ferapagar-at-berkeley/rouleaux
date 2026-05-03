import torch
import torchvision.io as io
import matplotlib.pyplot as plt
import glob
import matplotlib.colors as mcolors
import numpy as np
import os
from PIL import Image as PILImage
from matplotlib.patches import Patch

def generate_random_colors(n: int = 1):
    hues = np.random.rand(n)
    sats = 0.5 + 0.5 * np.random.rand(n)
    vals = 0.5 + 0.5 * np.random.rand(n)
    colors = [
        [int(c * 255) for c in mcolors.hsv_to_rgb([h, s, v])]
        for h, s, v in zip(hues, sats, vals)
    ]
    return colors[0] if n == 1 else colors

class Image:

    def __init__(self, data: torch.Tensor, title: str="", filter_name="original", fmt="rgb", palette: dict = None, legend_labels: dict = None):
        # Format input to [3, H, W]
        if data.dim() == 3 and data.shape[-1] == 3 and data.shape[0] != 3:
            data = data.permute(2, 0, 1)
        
        self.title = title
        self.data = data
        self.width = data.shape[-1]
        self.height = data.shape[-2]
        self.filter = filter_name
        self.format = fmt
        self.palette = palette
        self.legend_labels = legend_labels
        self.versions = {}  # Per-image version dictionary

    @classmethod
    def from_path(cls, path: str):
        data = io.read_image(path)
        return cls(data, title=path.split('\\')[-1].split('/')[-1].split('.')[0])

    @classmethod
    def from_dir(cls, dirs) -> list["Image"]:
        if isinstance(dirs, str):
            dirs = [dirs]
        imgs = []
        for path in dirs:
            for img_path in glob.glob(f'{path}/*.jpg') + glob.glob(f'{path}/*.png'):
                imgs.append(cls.from_path(img_path))
        return imgs

    @classmethod
    def slice_images(cls, images: list["Image"], slice_width: int) -> list["Image"]:
        slices = []
        for img in images:
            h, w = img.height, img.width
            for y in range(0, h, slice_width):
                for x in range(0, w, slice_width):
                    y_end = min(y + slice_width, h)
                    x_end = min(x + slice_width, w)
                    
                    # Ensure square slices by skipping partial patches at edges
                    if (y_end - y) != slice_width or (x_end - x) != slice_width:
                        continue
                        
                    slice_data = img.data[..., y:y_end, x:x_end]
                    new_title = f"{img.title}_{x}_{y}"
                    
                    new_img = cls(
                        slice_data,
                        title=new_title,
                        filter_name=img.filter,
                        fmt=img.format,
                        palette=img.palette,
                        legend_labels=img.legend_labels
                    )
                    slices.append(new_img)
        return slices

    def add_version(self, processor_name: str, data: torch.Tensor, fmt="rgb", palette=None, legend_labels=None) -> "Image":
        if self.filter == "original":
            new_filter_name = processor_name
        else:
            new_filter_name = f"{self.filter}_{processor_name}"
            
        new_image = Image(data, title=self.title, filter_name=new_filter_name, fmt=fmt, palette=palette, legend_labels=legend_labels)
        self.versions[new_filter_name] = new_image
        new_image.versions = self.versions
        return new_image

    def render(self) -> np.ndarray:
        """
        Processes the internal tensor data and returns a ready-to-display uint8 numpy array.
        """
        disp_data = self.data.clone().float()
        
        if self.format == "rgb":
            disp_data = disp_data.permute(1, 2, 0)
            disp_data = torch.clamp(disp_data, 0, 255).byte().numpy()
            return disp_data
            
        elif self.format in ["grayscale", "bw"]:
            if disp_data.dim() == 3 and disp_data.shape[0] == 3:
                disp_data = disp_data.mean(dim=0)
            elif disp_data.dim() == 3 and disp_data.shape[0] == 1:
                disp_data = disp_data.squeeze(0)
            
            if disp_data.max() <= 1.0:
                disp_data = disp_data * 255
            disp_data = torch.clamp(disp_data, 0, 255).byte().numpy()
            return disp_data
            
        elif self.format == "categorical":
            if disp_data.dim() == 3:
                mask = disp_data[0]
            else:
                mask = disp_data
                
            H, W = mask.shape
            rgb_img = np.zeros((H, W, 3), dtype=np.uint8)
            
            if self.palette is None:
                self.palette = {}
                
            unique_vals = torch.unique(mask).detach().cpu().numpy()
            
            for int_val in unique_vals:
                int_val = int(int_val)
                if int_val == 0:
                    continue
                    
                if int_val not in self.palette:
                    self.palette[int_val] = generate_random_colors(1)
                    
                rgb_color = self.palette[int_val]
                region = (mask == int_val).detach().cpu().numpy()
                rgb_img[region] = rgb_color
                
            return rgb_img
            
        return disp_data.byte().numpy()

    def show(self, version: str = None, ax=None, figsize=None):
        
        if version is None:
            img_to_show = self
        else:
            if version in self.versions:
                img_to_show = self.versions[version]
            else:
                raise ValueError(f"Version '{version}' not found.")
                
        if ax is None:
            plt.figure(figsize=figsize)
            ax = plt.gca()

        rendered_img = img_to_show.render()
        
        if img_to_show.format in ["grayscale", "bw"]:
            ax.imshow(rendered_img, cmap='gray', vmin=0, vmax=255)
        else:
            ax.imshow(rendered_img)
            
        if img_to_show.format == "categorical" and getattr(img_to_show, 'legend_labels', None) is not None:
            legend_elements = []
            for val, label in img_to_show.legend_labels.items():
                val_int = int(val)
                if val_int in img_to_show.palette:
                    color = np.array(img_to_show.palette[val_int]) / 255.0
                    legend_elements.append(Patch(facecolor=color, label=label))
            if legend_elements:
                ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

        ax.set_title(img_to_show.filter)
        ax.axis('off')

    def save(self, directory: str, version: str = None) -> str:
        
        if version is None:
            img_to_save = self
        else:
            if version in self.versions:
                img_to_save = self.versions[version]
            else:
                raise ValueError(f"Version '{version}' not found.")
                
        os.makedirs(directory, exist_ok=True)
        rendered_img = img_to_save.render()
        
        filename = f"{img_to_save.title}_{img_to_save.filter}.jpg" \
            if img_to_save.filter != "original" else f"{img_to_save.title}.jpg"

        filepath = os.path.join(directory, filename)

        pil_img = PILImage.fromarray(rendered_img)
        pil_img.save(filepath)
        
        return filepath

    @classmethod
    def save_images(cls, images: list["Image"], directory: str, version: str = None) -> list[str]:
        """
        Saves a list of images to the specified directory.
        """
        return [img.save(directory, version) for img in images]

    def show_versions(self, versions: list[str] = None, figsize=None, n_cols: int = None):
        """
        Plots multiple versions of the image in a grid.
        """
        if versions is None:
            versions = ["original"] + list(self.versions.keys())
        
        n = len(versions)
        if n == 0:
            print("No versions to show.")
            return
            
        if n_cols is None:
            cols = 2 if n >= 2 else 1
        else:
            cols = n_cols
            
        rows = (n + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize or (cols * 6, rows * 6))
        
        if self.title:
            fig.suptitle(self.title, fontsize=16)
        
        # Flatten axes for consistent indexing
        if n == 1 and cols == 1:
            axes_list = [axes]
        else:
            axes_list = axes.flatten()
            
        for i, version in enumerate(versions):
            if version == "original":
                self.show(ax=axes_list[i])
            else:
                self.show(version=version, ax=axes_list[i])
                
        # Hide remaining axes if the grid is larger than the number of versions
        for j in range(i + 1, len(axes_list)):
            axes_list[j].axis('off')
            
        plt.tight_layout()
