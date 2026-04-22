"""
Specialized detection for uppercase letters, focusing on B which is commonly misrecognized
"""

import cv2
import numpy as np
import os
from scipy import signal

def detect_letter_b(image):
    """
    Specialized detector for the letter B based on structural analysis
    
    Args:
        image: Input image containing a potential B character
        
    Returns:
        tuple: (is_b, confidence) - Boolean indicating if it's a B and confidence score
    """
    # Convert to grayscale if needed
    if len(image.shape) > 2 and image.shape[2] > 1:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Make sure image is uint8
    if np.max(gray) <= 1.0:
        gray = (gray * 255).astype(np.uint8)
    
    # Ensure proper size for analysis
    if gray.shape[0] < 20 or gray.shape[1] < 20:
        gray = cv2.resize(gray, (28, 28))
    elif max(gray.shape) > 200:
        # Scale down very large images for more consistent processing
        scale_factor = 200 / max(gray.shape)
        new_width = int(gray.shape[1] * scale_factor)
        new_height = int(gray.shape[0] * scale_factor)
        gray = cv2.resize(gray, (new_width, new_height))
    
    # Apply preprocessing to enhance features
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding for better separation
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Clean up noise with morphological operations
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Save preprocessed image for debugging
    debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_output")
    os.makedirs(debug_dir, exist_ok=True)
    cv2.imwrite(os.path.join(debug_dir, "b_detection_specialized.png"), binary)
    
    # Simple approach for small images: direct pixel analysis
    # For the specific case of letter B, examine the pattern directly
    h, w = binary.shape
    
    # Create regions for analysis
    left_strip = binary[:, :w//4]
    center_strip = binary[:, w//4:3*w//4]
    right_strip = binary[:, 3*w//4:]
    
    top_half = binary[:h//2, :]
    bottom_half = binary[h//2:, :]
    
    # Calculate white pixel ratios in each region
    total_pixels = h * w
    left_ratio = np.sum(left_strip > 0) / (h * w/4) if h*w > 0 else 0
    center_ratio = np.sum(center_strip > 0) / (h * w/2) if h*w > 0 else 0
    right_ratio = np.sum(right_strip > 0) / (h * w/4) if h*w > 0 else 0
    
    top_ratio = np.sum(top_half > 0) / (total_pixels/2) if total_pixels > 0 else 0
    bottom_ratio = np.sum(bottom_half > 0) / (total_pixels/2) if total_pixels > 0 else 0
    
    # For B: Strong left edge, moderate center, and roughly equal top/bottom
    vertical_pattern_score = 0
    
    # Analyze vertical pattern (looking for a strong continuous line on the left)
    left_col_heights = []
    for i in range(w//4):
        col_sum = np.sum(binary[:, i] > 0)
        left_col_heights.append(col_sum / h if h > 0 else 0)
    
    # A strong vertical line would have high values across multiple columns
    vertical_strength = np.mean(left_col_heights) if left_col_heights else 0
    
    # Look for the characteristic "B" shape
    b_score = 0
    
    # Get contours for analysis
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Analyze contour characteristics if present
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Calculate aspect ratio (width to height)
        aspect_ratio = w / h if h > 0 else 0
        
        # Calculate circularity
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # B typically has a specific aspect ratio
        if 0.5 < aspect_ratio < 0.8:
            b_score += 0.15
            print(f"B aspect ratio score: +0.15 (ratio={aspect_ratio:.2f})")
        
        # B has moderate circularity
        if 0.4 < circularity < 0.7:
            b_score += 0.1
            print(f"B circularity score: +0.1 (circularity={circularity:.2f})")
            
        # Create a mask for the contour to analyze regions
        mask = np.zeros_like(binary)
        cv2.drawContours(mask, [largest_contour], 0, 255, -1)
        
        # Divide the mask into regions for analysis
        left_side = mask[:, :w//2]
        right_side = mask[:, w//2:]
        left_count = np.count_nonzero(left_side)
        right_count = np.count_nonzero(right_side)
        left_right_ratio = left_count / right_count if right_count > 0 else 999
        
        # B usually has a more filled left side
        if left_right_ratio > 0.9:
            b_score += 0.15
            print(f"B left/right distribution score: +0.15 (ratio={left_right_ratio:.2f})")
    
    # 1. B has a strong left edge
    if left_ratio > 0.4:
        b_score += 0.2
        print(f"B left edge score: +0.2 (ratio={left_ratio:.2f})")
    
    # 2. B has balanced top and bottom halves
    if abs(top_ratio - bottom_ratio) < 0.3:
        b_score += 0.15
        print(f"B top/bottom balance score: +0.15 (diff={abs(top_ratio - bottom_ratio):.2f})")
    
    # 3. B's center has moderate density
    if 0.15 < center_ratio < 0.7:
        b_score += 0.15
        print(f"B center density score: +0.15 (density={center_ratio:.2f})")
        
    # 4. B's right edge typically has less content than left
    if right_ratio < left_ratio:
        b_score += 0.1
        print(f"B edge comparison score: +0.1")
    
    # 5. Strong vertical component (key feature of B)
    if vertical_strength > 0.3:  # Lowered threshold to be more lenient
        vertical_score = min(0.3, vertical_strength * 0.5)
        b_score += vertical_score
        print(f"B vertical strength score: +{vertical_score:.2f} (strength={vertical_strength:.2f})")
    
    # 6. Check for two "bulges" in the horizontal profile (characteristic of B)
    # Create horizontal projection
    h_proj = np.sum(binary > 0, axis=1) / w if w > 0 else np.zeros(h)
    
    # Count significant peaks in the horizontal projection
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(h_proj, height=0.2, distance=h/5)  # Lowered threshold and distance
    
    # B typically has two peaks (upper and lower curves)
    if len(peaks) >= 2:  # Changed to >= 2
        b_score += 0.2
        print(f"B horizontal profile score: +0.2 (peaks={len(peaks)})")
        
    # 7. Check for density patterns typical of a B
    # Divide the image into a 3x3 grid
    h_third, w_third = h//3, w//3
    grid = [
        [binary[:h_third, :w_third], binary[:h_third, w_third:2*w_third], binary[:h_third, 2*w_third:]],
        [binary[h_third:2*h_third, :w_third], binary[h_third:2*h_third, w_third:2*w_third], binary[h_third:2*h_third, 2*w_third:]],
        [binary[2*h_third:, :w_third], binary[2*h_third:, w_third:2*w_third], binary[2*h_third:, 2*w_third:]]
    ]
    
    # Calculate the density of each grid cell
    densities = [[np.mean(cell > 0) for cell in row] for row in grid]
    
    # B typically has a strong left column and moderate middle column
    left_col_density = (densities[0][0] + densities[1][0] + densities[2][0]) / 3
    middle_col_density = (densities[0][1] + densities[1][1] + densities[2][1]) / 3
    
    if left_col_density > 0.3:  # Strong left column
        b_score += 0.1
        print(f"B left column density score: +0.1 (density={left_col_density:.2f})")
        
    if 0.1 < middle_col_density < 0.6:  # Moderate middle column
        b_score += 0.1
        print(f"B middle column density score: +0.1 (density={middle_col_density:.2f})")
        
    # 8. Check if it's manually drawn 'B' (thick strokes)
    # This would typically have significant white pixels
    strong_signal = np.mean(binary > 0) > 0.15  # Lowered threshold
    if strong_signal:
        b_score += 0.1
        print(f"B stroke density score: +0.1 (density={np.mean(binary > 0):.2f})")
    
    # For added robustness, use ORB feature matching with a template B if available
    try:
        template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                     "debug_output", "b_template.png")
        if os.path.exists(template_path):
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if template is not None:
                # Resize template to match input
                template = cv2.resize(template, (gray.shape[1], gray.shape[0]))
                
                # Create ORB detector
                orb = cv2.ORB_create(nfeatures=100)
                
                # Find keypoints and descriptors
                kp1, des1 = orb.detectAndCompute(binary, None)
                kp2, des2 = orb.detectAndCompute(template, None)
                
                if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
                    # Create matcher
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    
                    # Match descriptors
                    matches = bf.match(des1, des2)
                    
                    # Sort by distance
                    matches = sorted(matches, key=lambda x: x.distance)
                    
                    # If we have good matches, boost the score
                    if len(matches) > 5:
                        match_quality = 1.0 - min(1.0, np.mean([m.distance for m in matches[:5]]) / 100)
                        b_score += match_quality * 0.2
    except Exception as e:
        print(f"Template matching error: {e}")
    
    # Add additional checks for air writing detection
    
    # Check for image quality and add bonuses for clear B shapes
    if b_score >= 0.3:  # If we already have a moderate B score
        # Check for distinct left vertical edge (crucial for B)
        left_edge_profile = np.sum(binary[:, :3], axis=1)  # Sum first few columns
        vertical_coverage = np.sum(left_edge_profile > 0) / h if h > 0 else 0
        
        if vertical_coverage > 0.5:  # If vertical line covers more than half the height
            vertical_bonus = min(0.2, vertical_coverage * 0.3)
            b_score += vertical_bonus
            print(f"B vertical edge bonus: +{vertical_bonus:.2f} (coverage={vertical_coverage:.2f})")
    
        # Check for two distinct horizontal bulges (top and bottom curves of B)
        # Analyze the right half of the image for curves
        right_half = binary[:, w//2:]
        top_right = right_half[:h//2, :]
        bottom_right = right_half[h//2:, :]
        
        top_right_ratio = np.sum(top_right > 0) / (top_right.size) if top_right.size > 0 else 0
        bottom_right_ratio = np.sum(bottom_right > 0) / (bottom_right.size) if bottom_right.size > 0 else 0
        
        # Both top and bottom curves should have reasonable presence
        if top_right_ratio > 0.15 and bottom_right_ratio > 0.15:
            curve_bonus = min(0.25, (top_right_ratio + bottom_right_ratio) * 0.5)
            b_score += curve_bonus
            print(f"B curves bonus: +{curve_bonus:.2f} (top={top_right_ratio:.2f}, bottom={bottom_right_ratio:.2f})")
    
    # Special case for air writing: if we have a clear vertical line and some right side content
    # This helps with quickly drawn Bs that might be less defined
    if left_ratio > 0.4 and right_ratio > 0.1 and vertical_strength > 0.3:
        b_score += 0.15
        print("Air writing B pattern bonus: +0.15")
    
    # Limit the score to 0.95
    b_score = min(b_score, 0.95)
    
    # Debug output
    print(f"B detection - Left: {left_ratio:.2f}, Center: {center_ratio:.2f}, Right: {right_ratio:.2f}, " +
          f"Top/Bottom: {top_ratio:.2f}/{bottom_ratio:.2f}, Vertical: {vertical_strength:.2f}, " +
          f"Score: {b_score:.2f}")
    
    # For debug purposes, create a directory and save analyzed images
    try:
        debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_output")
        os.makedirs(debug_dir, exist_ok=True)
        
        # Save the binary image
        cv2.imwrite(os.path.join(debug_dir, "b_detection_binary.png"), binary)
        
        # Create visualization of the regions - ensure all regions have the same dimensions
        viz = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Create each colored channel separately with proper broadcasting
        blue_channel = np.zeros((h, w), dtype=np.uint8)
        blue_channel[:, :w//4] = left_strip
        
        green_channel = np.zeros((h, w), dtype=np.uint8)
        green_channel[:, w//4:3*w//4] = center_strip
        
        red_channel = np.zeros((h, w), dtype=np.uint8)
        red_channel[:, 3*w//4:] = right_strip
        
        # Combine channels safely
        viz[:, :, 0] = blue_channel  # Blue channel
        viz[:, :, 1] = green_channel  # Green channel
        viz[:, :, 2] = red_channel  # Red channel
        
        # Save the visualization
        cv2.imwrite(os.path.join(debug_dir, "b_detection_regions.png"), viz)
    except Exception as e:
        print(f"Debug output error: {e}")
    
    # Return True if the score exceeds threshold
    # Using a lower threshold to be more lenient
    is_b = b_score > 0.35  # Lowered threshold even further
    
    print(f"Final B detection score: {b_score:.2f}, is_b={is_b}")
    return is_b, b_score

# Add additional specialized detectors as needed
letter_detectors = {
    'B': detect_letter_b,
    # Add more specialized detectors for other problematic letters
}

def detect_specific_letter(image, letter):
    """
    Apply a specialized detector for a specific letter
    
    Args:
        image: Input image
        letter: The letter to detect
        
    Returns:
        tuple: (is_match, confidence) - Boolean indicating if it matches and confidence score
    """
    if letter in letter_detectors:
        return letter_detectors[letter](image)
    
    # Default if no specialized detector exists
    return False, 0.0