"""Pattern computation utilities"""
import cv2
import numpy as np

def compute_all_patterns(image_path):
    """Returns [bg_complexity, edge_clarity, lighting, texture]"""
    img = cv2.imread(image_path)
    if img is None:
        return [0.5, 0.5, 0.5, 0.5]
    
    h, w = img.shape[:2]
    
    # Background complexity
    border = np.concatenate([img[:h//5, :].flatten(), img[-h//5:, :].flatten()])
    bg = min(np.std(border) / 60.0, 1.0)
    
    # Edge clarity
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge = min(np.mean(edges > 0) * 8, 1.0)
    
    # Lighting
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    light = 1.0 - min(np.std(hsv[:,:,2]) / 80.0, 1.0)
    
    # Texture
    patches = [np.std(gray[i:i+32, j:j+32]) for i in range(0, h-32, 32) for j in range(0, w-32, 32)]
    texture = 1.0 - min(np.std(patches) / 30.0, 1.0) if patches else 0.5
    
    return [bg, edge, light, texture]
