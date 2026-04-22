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

def distinguish_6_vs_9(image, initial_prediction):
    """
    Analyze if an image might be a 6 or 9, which are commonly confused due to rotation.
    
    Args:
        image: Input image (grayscale, 28x28)
        initial_prediction: Initial model prediction ('6' or '9')
        
    Returns:
        str: Corrected digit ('6' or '9') or initial prediction if unsure
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
    binary_uint8 = (binary * 255).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(binary_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return initial_prediction
    
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate moments
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return initial_prediction
    
    # Get centroid
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    # Divide image into quadrants relative to centroid
    h, w = binary.shape
    top_half = binary_uint8[0:cy, :]
    bottom_half = binary_uint8[cy:h, :]
    
    # Get mass distribution
    top_mass = np.sum(top_half)
    bottom_mass = np.sum(bottom_half)
    
    # Calculate circularity (perimeter^2 / area)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    circularity = (perimeter**2) / (4 * np.pi * area) if area > 0 else 0
    
    # For digit 6: loop is at the bottom
    # For digit 9: loop is at the top
    if initial_prediction == '6':
        # If more mass at the top than bottom, it might be a 9
        if top_mass > bottom_mass * 1.2:
            return '9'
    elif initial_prediction == '9':
        # If more mass at the bottom than top, it might be a 6
        if bottom_mass > top_mass * 1.2:
            return '6'
    
    # Additional check using holes position
    mask = np.zeros_like(binary_uint8)
    cv2.drawContours(mask, [largest_contour], 0, 255, -1)
    inverted = 255 - mask
    
    # Find holes
    hole_contours, _ = cv2.findContours(inverted, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    # If we found holes, check their position
    if hole_contours:
        significant_holes = [cnt for cnt in hole_contours if cv2.contourArea(cnt) > 5]
        if significant_holes:
            # Find the largest hole
            largest_hole = max(significant_holes, key=cv2.contourArea)
            hM = cv2.moments(largest_hole)
            if hM["m00"] > 0:
                # Get hole centroid
                hcx = int(hM["m10"] / hM["m00"])
                hcy = int(hM["m01"] / hM["m00"])
                
                # If hole is in top half but prediction is 6, change to 9
                if hcy < cy and initial_prediction == '6':
                    return '9'
                # If hole is in bottom half but prediction is 9, change to 6
                elif hcy > cy and initial_prediction == '9':
                    return '6'
    
    # If no clear evidence to change, return initial prediction
    return initial_prediction

def distinguish_0_vs_6(image, initial_prediction):
    """
    Distinguish between digits 0 and 6, which can be confused.
    
    Args:
        image: Input image (grayscale, 28x28)
        initial_prediction: Initial model prediction ('0' or '6')
        
    Returns:
        str: Corrected digit ('0' or '6') or initial prediction if unsure
    """
    # Ensure image is in correct format
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Normalize if needed
    if np.max(gray) > 1.0:
        gray = gray.astype(np.float32) / 255.0
    
    # Threshold the image
    _, binary = cv2.threshold(gray, 0.3, 1.0, cv2.THRESH_BINARY)
    binary_uint8 = (binary * 255).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(binary_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return initial_prediction
    
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate circularity (perimeter^2 / area)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    circularity = (perimeter**2) / (4 * np.pi * area) if area > 0 else 0
    
    # A "0" is typically more circular than a "6"
    if circularity < 1.3:  # Very circular shape suggests "0"
        return '0'
    
    # Create mask and check for top-bottom symmetry
    mask = np.zeros_like(binary_uint8)
    cv2.drawContours(mask, [largest_contour], 0, 255, -1)
    
    # Calculate moments
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return initial_prediction
    
    # Get centroid
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    # Divide image into top and bottom halves
    h, w = mask.shape
    top_half = mask[0:cy, :]
    bottom_half = mask[cy:h, :]
    
    # Analyze top-bottom distribution
    top_pixels = np.sum(top_half)
    bottom_pixels = np.sum(bottom_half)
    top_bottom_ratio = top_pixels / bottom_pixels if bottom_pixels > 0 else float('inf')
    
    # For "0", the ratio should be closer to 1 (more symmetric)
    # For "6", bottom half typically has more mass
    if top_bottom_ratio > 0.8 and top_bottom_ratio < 1.2:
        return '0'  # Balanced top and bottom suggests "0"
    elif top_bottom_ratio < 0.7:
        return '6'  # More mass in bottom suggests "6"
    
    return initial_prediction

def distinguish_1_vs_7(image, initial_prediction):
    """
    Distinguish between digits 1 and 7, which can be confused.
    
    Args:
        image: Input image (grayscale, 28x28)
        initial_prediction: Initial model prediction ('1' or '7')
        
    Returns:
        str: Corrected digit ('1' or '7') or initial prediction if unsure
    """
    # Ensure image is in correct format
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Normalize if needed
    if np.max(gray) > 1.0:
        gray = gray.astype(np.float32) / 255.0
    
    # Threshold the image
    _, binary = cv2.threshold(gray, 0.3, 1.0, cv2.THRESH_BINARY)
    binary_uint8 = (binary * 255).astype(np.uint8)
    
    # Horizontal projection histogram (sum across rows)
    horizontal_projection = np.sum(binary_uint8, axis=1)
    
    # For digit 7, there should be significant mass in the top portion
    h = binary_uint8.shape[0]
    top_third_mass = np.sum(horizontal_projection[:h//3])
    middle_third_mass = np.sum(horizontal_projection[h//3:2*h//3])
    
    # 7 typically has more mass in the top third compared to middle third
    if top_third_mass > middle_third_mass * 1.5:
        return '7'
    
    # Check for horizontal stroke at top (characteristic of 7)
    top_quarter = binary_uint8[:h//4, :]
    horizontal_sum_top = np.sum(top_quarter, axis=0)
    horizontal_mass_top = np.sum(horizontal_sum_top > 0)
    
    # If horizontal line at top spans more than 40% of width, likely a 7
    if horizontal_mass_top > binary_uint8.shape[1] * 0.4:
        return '7'
    
    # Check vertical profile - digit 1 is primarily vertical
    vertical_projection = np.sum(binary_uint8, axis=0)
    non_zero_cols = np.count_nonzero(vertical_projection)
    total_cols = binary_uint8.shape[1]
    
    # If less than 25% of columns have pixels, likely a thin vertical 1
    if non_zero_cols < total_cols * 0.25:
        return '1'
    
    return initial_prediction

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

def enhance_digit_features(image, digit_pair=None):
    """
    General-purpose enhancement for digit features
    
    Args:
        image: Input image (grayscale, 28x28)
        digit_pair: Optional pair of digits to target specific enhancements
        
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
    
    # Apply base enhancements
    # 1. Increase contrast
    enhanced = np.power(gray, 0.85)  # Gamma correction to enhance features
    
    # 2. Apply a slight blur to connect potentially broken strokes
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # For specific digit pairs, add targeted enhancements
    if digit_pair == '69':
        # For 6 vs 9: enhance orientation features
        enhanced_uint8 = (enhanced * 255).astype(np.uint8)
        # Use morphological operations to enhance the loop shape
        kernel = np.ones((2, 2), np.uint8)
        enhanced_uint8 = cv2.morphologyEx(enhanced_uint8, cv2.MORPH_CLOSE, kernel)
        enhanced = enhanced_uint8.astype(np.float32) / 255.0
    
    elif digit_pair == '06':
        # For 0 vs 6: enhance circularity and closure
        enhanced_uint8 = (enhanced * 255).astype(np.uint8)
        # Close gaps to better define the circular shape
        kernel = np.ones((2, 2), np.uint8)
        enhanced_uint8 = cv2.morphologyEx(enhanced_uint8, cv2.MORPH_CLOSE, kernel)
        enhanced = enhanced_uint8.astype(np.float32) / 255.0
    
    elif digit_pair == '17':
        # For 1 vs 7: enhance horizontal lines
        enhanced_uint8 = (enhanced * 255).astype(np.uint8)
        # Use horizontal kernel to enhance horizontal strokes (top of 7)
        kernel = np.ones((1, 3), np.uint8)
        enhanced_uint8 = cv2.morphologyEx(enhanced_uint8, cv2.MORPH_DILATE, kernel)
        enhanced = enhanced_uint8.astype(np.float32) / 255.0
    
    # Reshape for model compatibility if needed
    if len(image.shape) > 2 and image.shape[2] > 1:
        enhanced = enhanced.reshape(enhanced.shape[0], enhanced.shape[1], 1)
    
    return enhanced