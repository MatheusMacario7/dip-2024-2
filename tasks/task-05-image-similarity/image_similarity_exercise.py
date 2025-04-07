# image_similarity_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `compare_images(i1, i2)` that receives two grayscale images
represented as NumPy arrays (2D arrays of shape (H, W)) and returns a dictionary with the following metrics:

1. Mean Squared Error (MSE)
2. Peak Signal-to-Noise Ratio (PSNR)
3. Structural Similarity Index (SSIM) - simplified version without using external libraries
4. Normalized Pearson Correlation Coefficient (NPCC)

You must implement these functions yourself using only NumPy (no OpenCV, skimage, etc).

Each function should be implemented as a helper function and called inside `compare_images(i1, i2)`.

Function signature:
    def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:

The return value should be like:
{
    "mse": float,
    "psnr": float,
    "ssim": float,
    "npcc": float
}

Assume that i1 and i2 are normalized grayscale images (values between 0 and 1).
"""

import numpy as np

def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:

    def calc_mse(i1, i2):
        return np.mean((i1 - i2) ** 2)
    
    def calc_psnr(mse_val):
        if mse_val == 0:
            return float('inf')
        return 10 * np.log10(1.0 / mse_val)
    
    def calc_ssim(i1, i2):
        mu1 = np.mean(i1)
        mu2 = np.mean(i2)
        var1 = np.var(i1)
        var2 = np.var(i2)
        covar = np.mean((i1 - mu1) * (i2 - mu2))
        c1 = (0.01) ** 2
        c2 = (0.03) ** 2
        numerator = (2 * mu1 * mu2 + c1) * (2 * covar + c2)
        denominator = (mu1**2 + mu2**2 + c1) * (var1 + var2 + c2)
        return numerator / denominator
    
    def calc_npcc(i1, i2):
        centered1 = i1 - np.mean(i1)
        centered2 = i2 - np.mean(i2)
        numerator = np.sum(centered1 * centered2)
        denominator = np.sqrt(np.sum(centered1 ** 2)) * np.sqrt(np.sum(centered2 ** 2))
        if denominator == 0:
            if np.array_equal(i1, i2):
                return 1.0
            else:
                return 0.0
        return numerator / denominator
    
    mse_val = calc_mse(i1, i2)
    psnr_val = calc_psnr(mse_val)
    ssim_val = calc_ssim(i1, i2)
    npcc_val = calc_npcc(i1, i2)
    
    return {
        "mse": float(mse_val),
        "psnr": float(psnr_val),
        "ssim": float(ssim_val),
        "npcc": float(npcc_val)
    }
    pass