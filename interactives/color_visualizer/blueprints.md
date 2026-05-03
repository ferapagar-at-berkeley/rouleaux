### Color selector visualizer

I want a small html page that helps me decide the best filter for classifying some colors. 
These are the features I want:
 - Image path: select the image to analyze.
 - Color palettes: select three different colors (call them A, B, C)
 - Triangle view: a canvas showing the three colors in the corners of a triangle. The triangle should be filled with the corresponding interpolation of the three colors. 
    - Segment points: there will be three draggable points, each one constrained on a vertex of the triangle. Call their colors A', B', C'. 
    - Moving point: there will be a fourth draggable point constrained to the interior of the triangle. Call this O
 - Three pixel processing functions:
    - Plane project: project each pixel to the plane defined by the three colors
    - Triangle project: project each pixel to the triangle defined by the three colors
    - Mask-plane: divide the plane into 3 regions: 
      - the one bounded by the rays OB' and OC' (and containing A)
      - the one bounded by the rays OC' and OA' (and containing B)
      - the one bounded by the rays OA' and OB' (and containing C)
      Send each pixel to the vertex color of the region it belongs to.
    - Mask-triangle: divide the triangle into 3 regions: AB'OC', BC'OA', CA'OB'. Send each pixel to the vertex color of the region it belongs to.
 - View selector: select which pixel processing function to choose for the image view (plus original)
 - Image view: another canvas showing the original image together with the edited image in the format indicated by the view selector
 
 

