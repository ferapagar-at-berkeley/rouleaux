import numpy as np
import json
import matplotlib.colors as mcolors
from abc import ABC, abstractmethod
from image import Image
from geometry import get_uv, project_to_triangle, project_to_plane, get_triangle_uv, mask_plane_labels

class Processor(ABC):
    def __init__(self, **kwargs):
        self.config = kwargs
        self.name = kwargs.get("name", self.__class__.__name__)
        self.fmt = kwargs.get("fmt", "rgb")
        self.legend_labels = kwargs.get("legend_labels", None)

    @abstractmethod
    def _process(self, image: Image) -> np.ndarray:
        pass

    def process(self, image: Image) -> Image:
        data = self._process(image)
        return image.add_version(
            self.name, data, fmt=self.fmt, 
            palette=getattr(self, 'palette', None),
            legend_labels=getattr(self, 'legend_labels', None)
        )

    @classmethod
    def from_json(cls, path: str, **kwargs):
        with open(path, 'r') as f:
            config = json.load(f)
        
        # Convert lists to tensors
        for key in config:
            if isinstance(config[key], list):
                config[key] = np.array(config[key])
        
        # Update config with kwargs
        config.update(kwargs)
        
        return cls(**config)


# ---------------------------- Plane Processors -----------------------------

class PlaneProjection2D(Processor):
    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, name = "plane_projection_2d", **kwargs):
        super().__init__(A=A, B=B, C=C, name=name, fmt="uv", **kwargs)
        self.A = A
        self.B = B
        self.C = C

    def _process(self, image: Image) -> np.ndarray:
        return get_uv(image.data, self.A, self.B, self.C)

class PlaneProjection(Processor):
    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, name = "plane_projection", **kwargs):
        super().__init__(A=A, B=B, C=C, name=name, fmt="rgb", **kwargs)
        self.A = A
        self.B = B
        self.C = C

    def _process(self, image: Image) -> np.ndarray:
        return project_to_plane(image.data, self.A, self.B, self.C)

class PlaneMask(Processor):
    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, 
                 v_O: np.ndarray, t_Ap: float, t_Bp: float, t_Cp: float, name="plane_mask", **kwargs):
        super().__init__(A=A, B=B, C=C, v_O=v_O, t_Ap=t_Ap, t_Bp=t_Bp, t_Cp=t_Cp, name=name, fmt="categorical", **kwargs)
        self.A = A
        self.B = B
        self.C = C
        self.v_O = v_O
        self.t_Ap = t_Ap
        self.t_Bp = t_Bp
        self.t_Cp = t_Cp
        self.fmt = "categorical"
        
        def to_rgb(t):
            return [int(x) for x in t]
            
        self.palette = {
            1: to_rgb(self.A),
            2: to_rgb(self.B),
            3: to_rgb(self.C)
        }

    def _process(self, image: Image) -> np.ndarray:
        uv = get_uv(image.data, self.A, self.B, self.C)
        labels = mask_plane_labels(uv, self.v_O, self.t_Ap, self.t_Bp, self.t_Cp)
        return np.expand_dims(labels, 0)


# ---------------------------- Triangle Processors ----------------------------

class TriangleProjection2D(Processor):
    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, name = "triangle_projection_2d", **kwargs):
        super().__init__(A=A, B=B, C=C, name=name, fmt="uv", **kwargs)
        self.A = A
        self.B = B
        self.C = C

    def _process(self, image: Image) -> np.ndarray:
        return get_triangle_uv(image.data, self.A, self.B, self.C)

class TriangleProjection(Processor):
    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, name = "triangle_projection", **kwargs):
        super().__init__(A=A, B=B, C=C, name=name, fmt="rgb", **kwargs)
        self.A = A
        self.B = B
        self.C = C

    def _process(self, image: Image) -> np.ndarray:
        return project_to_triangle(image.data, self.A, self.B, self.C)

class TriangleMask(Processor):
    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, 
                 v_O: np.ndarray, t_Ap: float, t_Bp: float, t_Cp: float, name="triangle_mask", **kwargs):
        super().__init__(A=A, B=B, C=C, v_O=v_O, t_Ap=t_Ap, t_Bp=t_Bp, t_Cp=t_Cp, name=name, fmt="categorical", **kwargs)
        self.A = A
        self.B = B
        self.C = C
        self.v_O = v_O
        self.t_Ap = t_Ap
        self.t_Bp = t_Bp
        self.t_Cp = t_Cp
        self.fmt = "categorical"
        
        def to_rgb(t):
            return [int(x) for x in t]
            
        self.palette = {
            1: to_rgb(self.A),
            2: to_rgb(self.B),
            3: to_rgb(self.C)
        }

    def _process(self, image: Image) -> np.ndarray:
        uv = get_triangle_uv(image.data, self.A, self.B, self.C)
        labels = mask_plane_labels(uv, self.v_O, self.t_Ap, self.t_Bp, self.t_Cp)
        return np.expand_dims(labels, 0)
