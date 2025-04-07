# image_geometry_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `apply_geometric_transformations(img)` that receives a grayscale image
represented as a NumPy array (2D array) and returns a dictionary with the following transformations:

1. Translated image (shift right and down)
2. Rotated image (90 degrees clockwise)
3. Horizontally stretched image (scale width by 1.5)
4. Horizontally mirrored image (flip along vertical axis)
5. Barrel distorted image (simple distortion using a radial function)

You must use only NumPy to implement these transformations. Do NOT use OpenCV, PIL, skimage or similar libraries.

Function signature:
    def apply_geometric_transformations(img: np.ndarray) -> dict:

The return value should be like:
{
    "translated": np.ndarray,
    "rotated": np.ndarray,
    "stretched": np.ndarray,
    "mirrored": np.ndarray,
    "distorted": np.ndarray
}
"""

import numpy as np

def apply_geometric_transformations(img: np.ndarray) -> dict:
    H, W = img.shape
    
    # Translated image (shift right by 10, down by 10)
    translated = np.zeros_like(img)
    shift_down, shift_right = 10, 10
    if H > shift_down and W > shift_right:
        translated[shift_down:, shift_right:] = img[:-shift_down, :-shift_right]
    else:
        translated = img.copy()
    
    # Rotated image (90 degrees clockwise)
    rotated = np.rot90(img, k=-1)
    
    # Horizontally stretched image (scale width by 1.5)
    new_width = int(W * 1.5)
    x_coords = np.linspace(0, W-1, num=new_width)
    indices = np.floor(x_coords).astype(int)
    stretched = img[:, indices]
    
    # Horizontally mirrored image
    mirrored = img[:, ::-1]
    
    # Barrel distortion
    rows, cols = np.indices((H, W))
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    dx = cols - cx
    dy = rows - cy
    r = np.sqrt(dx**2 + dy**2)
    max_r = np.sqrt(cx**2 + cy**2)
    r_norm = r / max_r
    
    a = -0.1  # Distortion coefficient
    distortion = 1 + a * r_norm**2
    new_dx = dx * distortion
    new_dy = dy * distortion
    
    new_x = np.clip(cx + new_dx, 0, W-1)
    new_y = np.clip(cy + new_dy, 0, H-1)
    
    x_indices = np.round(new_x).astype(int)
    y_indices = np.round(new_y).astype(int)
    
    distorted = img[y_indices, x_indices]
    
    return {
        "translated": translated,
        "rotated": rotated,
        "stretched": stretched,
        "mirrored": mirrored,
        "distorted": distorted
    }
    pass