

## image.py
Image: class for image handling:
- init takes a torch tensor of shape [3, H, W] or [H, W, 3]. Saves as [3, H, W]
- class method to create an image from a path
- class method to create a list of images from a (list of) directory(ies)
- additional attibutes: 
    - width
    - height
    - filter (default "original")
    - format (default "rgb", can be 'grayscale', 'bw' or 'categorical-N' where N is the number of categories)
    - versions (dict) stores versions of the image labeled by the processor that created them. This dict is shared across all instances of the Image class.
- add_version(self, processor_name: str, data: torch.Tensor) -> None: creates an Image instance with the processed data and adds it to the versions dict. If self.filter == 'origin', use the processor name as the dict key (and as the new Image's filter attribute). Otherwise, the name (and the new Image's filter attribute) will be self.filter+'_'+processor_name.
- show(self, version: str = None) -> None: shows the image of the specified version. If no version is specified, it shows the current image (it's filter attribute). Grayscale and bw will interpolate the interval [0,1] into [0,255] for display. Categorigal-N will turn each value 1,... N to a different hue in the color wheel (arranged in steps of 180/N-1 from 0 to 180).

## processor.py
Processor: Abstract base class for image processing:
 - init takes a **kwargs for config
 - abstract method _process(self, image: Image) -> Image (uses the processed parameters from config)
 - process(self, image: Image) -> Image: calls _process and adds the version to the image

2DPlaneProjection: Processor for projecting images onto the plane defined by three points A, B, C. Coordinates are the (u,v) that verify P' = A + u*(B-A) + v*(C-A).
 - config init takes three points A, B, C.
 - _process(self, image: Image) -> Image: projects the image onto the plane defined by A, B, C.
 
2DTriangleProjection: Processor for projecting images onto the triangle defined by three points A, B, C. Coordinates are the (u,v) that verify P' = A + u*(B-A) + v*(C-A).
 - config init takes three points A, B, C.
 - _process(self, image: Image) -> Image: projects the image onto the triangle defined by A, B, C by leveraging the 2DPlaneProjection processor and clipping (u,v) to the [(0,0), (0,1), (1,0)] triangle.

PlaneProjection: Processor for projecting images onto the plane defined by three points A, B, C.
 - config init takes three points A, B, C.
 - _process(self, image: Image) -> Image: projects the image onto the plane defined by A, B, C by leveraging the 2DPlaneProjection processor.
 
TriangleProjection: Processor for projecting images onto the triangle defined by three points A, B, C.
 - config init takes three points A, B, C.
 - _process(self, image: Image) -> Image: projects the image onto the triangle defined by A, B, C by leveraging the 2DTriangleProjection processor.


## triangle_mask.py
TriangleMask: Processor for masking images to 1,2,3.
 - config takes:
    - three points A, B, C defining a triangle
    - three values t_A', t_B', t_C' in [0,1] that define some interpolated points A', B', C' on the edges of the triangle (A' on BC, B' on AC, C' on AB).
    - A 3D vector v_O of entries that add to 1, which defines a point O= v_O^T * [A,B,C]^T.
 - _process(self, image: Image) -> Image: divide the triangle into 3 regions: AB'OC', BC'OA', CA'OB'. Send each pixel to 1, 2, or 3 depending on which region it belongs to.

TriangleMaskColor: Processor for masking images to the colors of the triangle vertices.
 - config takes:
    - three points A, B, C defining a triangle
    - three values t_A', t_B', t_C' in [0,1] that define some interpolated points A', B', C' on the edges of the triangle (A' on BC, B' on AC, C' on AB).
    - A 3D vector v_O of entries that add to 1, which defines a point O= v_O^T * [A,B,C]^T.
 - _process(self, image: Image) -> Image: divide the triangle into 3 regions: AB'OC', BC'OA', CA'OB'. Send each pixel to A, B, or C depending on which region it belongs to (leverage TriangleMask processor).

 PlaneMask: Processor for masking images to 1,2,3.
  - config takes:
    - three points A, B, C defining a plane
    - three values t_A', t_B', t_C' in [0,1] that define some interpolated points A', B', C' on the edges of the triangle (A' on BC, B' on AC, C' on AB).
    - A 3D vector v_O of entries that add to 1, which defines a point O= v_O^T * [A,B,C]^T.
  - _process(self, image: Image) -> Image: divide the plane into 3 regions: 
      - the one bounded by the rays OB' and OC' (and containing A)
      - the one bounded by the rays OC' and OA' (and containing B)
      - the one bounded by the rays OA' and OB' (and containing C)
    Send each pixel to 1, 2, or 3 depending on which region it belongs to.
  
PlaneMaskColor: Processor for masking images to the colors of the triangle vertices.
  - config takes:
    - three points A, B, C defining a plane
    - three values t_A', t_B', t_C' in [0,1] that define some interpolated points A', B', C' on the edges of the triangle (A' on BC, B' on AC, C' on AB).
    - A 3D vector v_O of entries that add to 1, which defines a point O= v_O^T * [A,B,C]^T.
  - _process(self, image: Image) -> Image: divide the plane into 3 regions: 
      - the one bounded by the rays OB' and OC' (and containing A)
      - the one bounded by the rays OC' and OA' (and containing B)
      - the one bounded by the rays OA' and OB' (and containing C)
    Send each pixel to A, B, or C depending on which region it belongs to (leverage PlaneMask processor).
