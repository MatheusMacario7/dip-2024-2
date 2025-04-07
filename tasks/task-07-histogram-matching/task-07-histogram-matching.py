# histogram_matching_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `match_histograms_rgb(source_img, reference_img)` that receives two RGB images
(as NumPy arrays with shape (H, W, 3)) and returns a new image where the histogram of each RGB channel 
from the source image is matched to the corresponding histogram of the reference image.

Your task:
- Read two RGB images: source and reference (they will be provided externally).
- Match the histograms of the source image to the reference image using all RGB channels.
- Return the matched image as a NumPy array (uint8)

Function signature:
    def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray

Return:
    - matched_img: NumPy array of the result image

Notes:
- Do NOT save or display the image in this function.
- Do NOT use OpenCV to apply the histogram match (only for loading images, if needed externally).
- You can assume the input images are already loaded and in RGB format (not BGR).
"""

import cv2 as cv
import numpy as np
import scikitimage as ski

def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray:
        matched_img = np.zeros_like(source_img)
        
        for channel in range(3):  # Processar cada canal R, G, B separadamente
            src_channel = source_img[:, :, channel]
            ref_channel = reference_img[:, :, channel]
            
            # Calcular histogramas
            hist_src, _ = np.histogram(src_channel.flatten(), bins=256, range=(0, 256))
            hist_ref, _ = np.histogram(ref_channel.flatten(), bins=256, range=(0, 256))
            
            # Calcular CDFs (Função de Distribuição Cumulativa)
            cdf_src = hist_src.cumsum()
            cdf_ref = hist_ref.cumsum()
            
            # Normalizar CDFs para 0-255
            cdf_src = (cdf_src - cdf_src.min()) * 255 / (cdf_src.max() - cdf_src.min() + 1e-8)
            cdf_src = cdf_src.astype(np.uint8)
            
            cdf_ref = (cdf_ref - cdf_ref.min()) * 255 / (cdf_ref.max() - cdf_ref.min() + 1e-8)
            cdf_ref = cdf_ref.astype(np.uint8)
            
            # Criar tabela de mapeamento
            lookup_table = np.zeros(256, dtype=np.uint8)
            for intensity in range(256):
                lookup_table[intensity] = np.argmax(cdf_ref >= cdf_src[intensity])
            
            # Aplicar mapeamento
            matched_img[:, :, channel] = lookup_table[src_channel]
    
        return matched_img