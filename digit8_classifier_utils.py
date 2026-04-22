"""
Specialized classifier utility for improving distinction between commonly confused digit pairs.
This module provides preprocessing and prediction functions for improved recognition:
- 3 vs 8 distinction
- 6 vs 9 distinction
- 0 vs 6 distinction
- 1 vs 7 distinction
"""

import numpy as np
import cv2

def is_potentially_digit_eight(image):
    """
    Analyze if an image classified as '3' might actually be an '8' drawn in two parts.
    
    Args:
        image: Input image (grayscale, 28x28)
        
    Returns:
        bool: True if the image could be an '8', False otherwise
    """
    # Ensure image is in correct format
    if len(image.shape) > 2:
        # Convert to grayscale if color
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Normalize if needed
    if np.max(gray) > 1.0:
        gray = gray.astype(np.float32) / 255.0
    
    # Threshold the image
    _, binary = cv2.threshold(gray, 0.3, 1.0, cv2.THRESH_BINARY)
    
    # Convert to uint8 for contour detection
    binary_uint8 = (binary * 255).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(binary_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return False
    
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate region properties
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = float(w) / h if h > 0 else 0
    
    # Create a mask to analyze upper and lower regions separately
    mask = np.zeros_like(binary_uint8)
    cv2.drawContours(mask, [largest_contour], 0, 255, -1)
    
    # Divide the image into upper and lower halves
    h, w = mask.shape
    upper_half = mask[0:h//2, :]
    lower_half = mask[h//2:h, :]
    
    # Count white pixels in each half
    upper_pixels = np.sum(upper_half > 0)
    lower_pixels = np.sum(lower_half > 0)
    
    # Calculate balance between upper and lower halves
    total_pixels = upper_pixels + lower_pixels
    if total_pixels == 0:
        return False
    
    upper_ratio = upper_pixels / total_pixels
    lower_ratio = lower_pixels / total_pixels
    
    # If there's good balance between upper and lower regions (characteristic of '8')
    # and the aspect ratio is close to 1 (typical for '8')
    is_balanced = 0.3 <= upper_ratio <= 0.7 and 0.3 <= lower_ratio <= 0.7
    is_square_like = 0.6 <= aspect_ratio <= 1.4
    
    # Calculate moments to check for symmetry
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return False
    
    # Get centroid
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    # Check horizontal symmetry (important for '8')
    left_half = mask[:, 0:cx]
    right_half = mask[:, cx:w]
    
    # Flip right half for comparison
    right_half_flipped = cv2.flip(right_half, 1)
    
    # Resize for comparison if necessary
    if left_half.shape[1] != right_half_flipped.shape[1]:
        # Choose the smaller width
        min_width = min(left_half.shape[1], right_half_flipped.shape[1])
        left_half = left_half[:, :min_width]
        right_half_flipped = right_half_flipped[:, :min_width]
    
    # Calculate symmetry score
    overlap = np.logical_and(left_half > 0, right_half_flipped > 0)
    union = np.logical_or(left_half > 0, right_half_flipped > 0)
    
    symmetry_score = np.sum(overlap) / np.sum(union) if np.sum(union) > 0 else 0
    
    # Check for holes (8 typically has two holes)
    # Invert the binary image for hole detection
    inverted = 255 - binary_uint8
    holes, _ = cv2.findContours(inverted, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out small noise holes
    valid_holes = [cnt for cnt in holes if cv2.contourArea(cnt) > 10]
    
    # Decision based on combined factors
    return ((is_balanced and is_square_like) or 
            (symmetry_score > 0.5) or 
            (len(valid_holes) >= 2))

def enhance_8_vs_3_features(image):
    """
    Enhance features to better distinguish between digits 8 and 3
    
    Args:
        image: Input image (grayscale, 28x28)
        
    Returns:
        Enhanced image
    """
    # Ensure image is in correct format
    if len(image.shape) > 2:
        # Convert to grayscale if color
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Normalize if needed
    if np.max(gray) > 1.0:
        gray = gray.astype(np.float32) / 255.0
    
    # Apply image enhancement techniques
    # 1. Increase contrast
    enhanced = np.power(gray, 0.8)  # Gamma correction to enhance features
    
    # 2. Apply a slight blur to connect potentially broken strokes
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # 3. Adaptive thresholding for better feature definition
    enhanced_uint8 = (enhanced * 255).astype(np.uint8)
    binary = cv2.adaptiveThreshold(
        enhanced_uint8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # 4. Morphological operations to close small gaps
    kernel = np.ones((2, 2), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 5. Convert back to normalized float format
    result = closed.astype(np.float32) / 255.0
    
    # Reshape for model compatibility if needed
    if len(image.shape) > 2 and image.shape[2] > 1:
        result = result.reshape(result.shape[0], result.shape[1], 1)
    
    return result