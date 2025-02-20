import cv2
import numpy as np

def linear_combination(img1, img2, a, b):
    
    return np.clip(a * img1 + b * img2, 0, 255).astype(np.uint8)

def on_trackbar_change(*args):
    """Callback function for trackbar changes"""
    global img1, img2, window_name
    
    try:
        # Get current trackbar positions
        a = cv2.getTrackbarPos('a', window_name) / 100.0
        b = cv2.getTrackbarPos('b', window_name) / 100.0
        
        # Calculate linear combination
        result = linear_combination(img1, img2, a, b)
        
        # Display result
        cv2.imshow(window_name, result)
    except cv2.error:
        pass  # Handle initialization state

# Main window name
window_name = 'Linear Combination'

# Load images
img1 = cv2.imread('Atividade/imagem-1.png', cv2.IMREAD_COLOR)
img2 = cv2.imread('Atividade/imagem-2.jpg', cv2.IMREAD_COLOR)

# Check if images are loaded successfully
if img1 is None or img2 is None:
    print("Error: Couldn't load one or both images")
    exit()

# Resize images to same size if they're different
if img1.shape != img2.shape:
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# Create window and wait for it to be initialized
cv2.namedWindow(window_name)
cv2.imshow(window_name, img1)  # Show initial image

# Create trackbars (values from 0 to 100 to represent 0.0 to 1.0)
cv2.createTrackbar('a', window_name, 50, 100, on_trackbar_change)
cv2.createTrackbar('b', window_name, 50, 100, on_trackbar_change)

# Initial render
cv2.waitKey(1)  # Give time for window to initialize
on_trackbar_change(0)

# Wait for key press
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        break

# Cleanup
cv2.destroyAllWindows()