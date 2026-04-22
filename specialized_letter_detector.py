"""
Specialized letter detection module for all uppercase letters A-Z
"""

import os
import sys
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import traceback

# Dictionary to store loaded models
LETTER_MODELS = {}

def load_letter_detector_model(letter='B'):
    """Load a specialized letter detector model"""
    global LETTER_MODELS
    
    # Check if the model is already loaded
    if letter in LETTER_MODELS and LETTER_MODELS[letter] is not None:
        return LETTER_MODELS[letter]
    
    # Define model path
    model_path = f"models/specialized/{letter.lower()}_detector.h5"
    
    # Check if the model exists
    if not os.path.exists(model_path):
        print(f"Letter {letter} detector model not found at {model_path}")
        return None
    
    try:
        # Load the model
        model = keras.models.load_model(model_path)
        LETTER_MODELS[letter] = model
        print(f"{letter} detector model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading {letter} detector model: {e}")
        traceback.print_exc()
        return None

def detect_specific_letter(image, letter='B', threshold=0.5):
    """Detect if an image contains a specific letter"""
    # Load the model if not already loaded
    model = load_letter_detector_model(letter)
    if model is None:
        return False, 0.0
    
    try:
        # Preprocess the image
        if len(image.shape) == 3 and image.shape[2] == 3:  # Color image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize to 28x28
        resized = cv2.resize(gray, (28, 28))
        
        # Apply adaptive threshold to handle varying brightness
        adaptive_thresh = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY_INV, 11, 2)
        
        # Normalize and reshape for prediction
        normalized = adaptive_thresh.astype('float32') / 255.0
        input_img = normalized.reshape(1, 28, 28, 1)
        
        # Predict
        prediction = model.predict(input_img, verbose=0)[0]
        is_letter = prediction[1] > threshold
        confidence = float(prediction[1])
        
        print(f"Letter {letter} detection: {is_letter} (confidence: {confidence:.4f})")
        
        return is_letter, confidence
    except Exception as e:
        print(f"Error detecting letter {letter}: {e}")
        traceback.print_exc()
        return False, 0.0

def detect_all_letters(image, min_confidence=0.5):
    """Run all letter detectors and return the best match"""
    best_letter = None
    best_confidence = 0.0
    letter_scores = {}
    
    # Try all letters A-Z
    for letter_code in range(ord('A'), ord('Z') + 1):
        letter = chr(letter_code)
        is_letter, confidence = detect_specific_letter(image, letter, threshold=min_confidence)
        letter_scores[letter] = confidence
        
        # Update best match
        if is_letter and confidence > best_confidence:
            best_letter = letter
            best_confidence = confidence
    
    # Special cases for commonly confused letters
    # B, D, P, R are often confused
    if best_letter in ['B', 'D', 'P', 'R']:
        # Run additional structural analysis
        is_letter, structural_score = perform_structural_analysis(image, best_letter)
        if is_letter:
            best_confidence = max(best_confidence, structural_score)
        else:
            # Try other candidates
            candidates = ['B', 'D', 'P', 'R']
            for candidate in candidates:
                if candidate != best_letter:
                    is_letter, structural_score = perform_structural_analysis(image, candidate)
                    if is_letter and structural_score > best_confidence:
                        best_letter = candidate
                        best_confidence = structural_score
    
    return best_letter, best_confidence, letter_scores

def perform_structural_analysis(image, letter):
    """Perform structural analysis for a specific letter"""
    if letter == 'B':
        return analyze_b_structure(image)
    elif letter == 'D':
        return analyze_d_structure(image)
    elif letter == 'P':
        return analyze_p_structure(image)
    elif letter == 'R':
        return analyze_r_structure(image)
    elif letter == 'O':
        return analyze_o_structure(image)
    elif letter == 'Q':
        return analyze_q_structure(image)
    elif letter == 'C':
        return analyze_c_structure(image)
    # Add more letter-specific structural analysis as needed
    return False, 0.0

def analyze_b_structure(image):
    """Analyze if the image has B-like structure"""
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False, 0.0
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Calculate aspect ratio (width/height)
        aspect_ratio = w / h if h > 0 else 0
        
        # B typically has aspect ratio around 0.5-0.7
        aspect_ratio_score = 0.0
        if 0.4 < aspect_ratio < 0.8:
            aspect_ratio_score = 0.3
        
        # Divide image into left, middle, and right regions
        left_region = binary[y:y+h, x:x+int(w*0.4)]
        middle_region = binary[y:y+h, x+int(w*0.4):x+int(w*0.6)]
        right_region = binary[y:y+h, x+int(w*0.6):x+w]
        
        # Calculate white pixel ratios in each region
        left_ratio = np.sum(left_region > 0) / (left_region.shape[0] * left_region.shape[1]) if left_region.size > 0 else 0
        middle_ratio = np.sum(middle_region > 0) / (middle_region.shape[0] * middle_region.shape[1]) if middle_region.size > 0 else 0
        right_ratio = np.sum(right_region > 0) / (right_region.shape[0] * right_region.shape[1]) if right_region.size > 0 else 0
        
        # B typically has high density on left (vertical line), medium in middle, and lower on right
        region_score = 0.0
        if left_ratio > 0.5:  # Strong vertical line on the left
            region_score += 0.2
        
        if middle_ratio > 0.3:  # Some density in middle for the curves
            region_score += 0.2
        
        if right_ratio < left_ratio:  # Right side less dense than left (curves only)
            region_score += 0.2
        
        # Check for two horizontal peaks for top and bottom curves
        horizontal_profile = np.sum(binary, axis=1)
        peaks = 0
        for i in range(1, len(horizontal_profile) - 1):
            if horizontal_profile[i] > horizontal_profile[i-1] and horizontal_profile[i] > horizontal_profile[i+1]:
                peaks += 1
        
        peak_score = 0.0
        if peaks >= 2:  # B typically has at least 2 horizontal peaks for the curves
            peak_score = 0.3
        
        # Combine scores
        total_score = aspect_ratio_score + region_score + peak_score
        
        # Normalize to [0,1]
        normalized_score = min(1.0, total_score)
        
        is_b = normalized_score > 0.6
        
        print(f"B structural analysis: score={normalized_score:.2f}, is_b={is_b}")
        
        return is_b, normalized_score
    except Exception as e:
        print(f"Error in B structural analysis: {e}")
        return False, 0.0

def analyze_d_structure(image):
    """Analyze if the image has D-like structure"""
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False, 0.0
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Calculate aspect ratio (width/height)
        aspect_ratio = w / h if h > 0 else 0
        
        # D typically has aspect ratio around 0.5-0.7
        aspect_ratio_score = 0.0
        if 0.4 < aspect_ratio < 0.8:
            aspect_ratio_score = 0.3
        
        # Divide image into left and right regions
        left_region = binary[y:y+h, x:x+int(w*0.3)]
        right_region = binary[y:y+h, x+int(w*0.3):x+w]
        
        # Calculate white pixel ratios in each region
        left_ratio = np.sum(left_region > 0) / (left_region.shape[0] * left_region.shape[1]) if left_region.size > 0 else 0
        right_ratio = np.sum(right_region > 0) / (right_region.shape[0] * right_region.shape[1]) if right_region.size > 0 else 0
        
        # D typically has high density on left (vertical line), and curved edge on right
        region_score = 0.0
        if left_ratio > 0.6:  # Strong vertical line on the left
            region_score += 0.3
        
        # Calculate vertical profile
        vertical_profile = np.sum(binary, axis=0)
        
        # D should have decreasing density from left to right
        profile_score = 0.0
        if np.argmax(vertical_profile) < w // 3:  # Maximum density on left side
            profile_score = 0.3
        
        # Combine scores
        total_score = aspect_ratio_score + region_score + profile_score
        
        # Normalize to [0,1]
        normalized_score = min(1.0, total_score)
        
        is_d = normalized_score > 0.6
        
        print(f"D structural analysis: score={normalized_score:.2f}, is_d={is_d}")
        
        return is_d, normalized_score
    except Exception as e:
        print(f"Error in D structural analysis: {e}")
        return False, 0.0

def analyze_p_structure(image):
    """Analyze if the image has P-like structure"""
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False, 0.0
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Calculate aspect ratio (width/height)
        aspect_ratio = w / h if h > 0 else 0
        
        # P typically has aspect ratio around 0.5-0.7
        aspect_ratio_score = 0.0
        if 0.4 < aspect_ratio < 0.8:
            aspect_ratio_score = 0.2
        
        # Divide image into top and bottom halves
        top_half = binary[y:y+int(h*0.5), x:x+w]
        bottom_half = binary[y+int(h*0.5):y+h, x:x+w]
        
        # Divide into left and right regions
        left_region = binary[y:y+h, x:x+int(w*0.3)]
        right_top_region = binary[y:y+int(h*0.5), x+int(w*0.3):x+w]
        right_bottom_region = binary[y+int(h*0.5):y+h, x+int(w*0.3):x+w]
        
        # Calculate white pixel ratios in each region
        left_ratio = np.sum(left_region > 0) / (left_region.shape[0] * left_region.shape[1]) if left_region.size > 0 else 0
        right_top_ratio = np.sum(right_top_region > 0) / (right_top_region.shape[0] * right_top_region.shape[1]) if right_top_region.size > 0 else 0
        right_bottom_ratio = np.sum(right_bottom_region > 0) / (right_bottom_region.shape[0] * right_bottom_region.shape[1]) if right_bottom_region.size > 0 else 0
        
        # P has strong vertical line on the left, curved region on top right, and empty bottom right
        region_score = 0.0
        if left_ratio > 0.6:  # Strong vertical line on the left
            region_score += 0.2
        
        if right_top_ratio > 0.3:  # Some density in top right for the curve
            region_score += 0.2
        
        if right_bottom_ratio < 0.2:  # Low density in bottom right (empty)
            region_score += 0.2
        
        # Calculate horizontal density profiles
        horizontal_profile = np.sum(binary, axis=1)
        top_density = np.sum(horizontal_profile[:h//2]) / (h//2) if h > 0 else 0
        bottom_density = np.sum(horizontal_profile[h//2:]) / (h - h//2) if h > 0 else 0
        
        # P has higher density in top half than bottom half
        density_score = 0.0
        if top_density > bottom_density:
            density_score = 0.2
        
        # Combine scores
        total_score = aspect_ratio_score + region_score + density_score
        
        # Normalize to [0,1]
        normalized_score = min(1.0, total_score)
        
        is_p = normalized_score > 0.6
        
        print(f"P structural analysis: score={normalized_score:.2f}, is_p={is_p}")
        
        return is_p, normalized_score
    except Exception as e:
        print(f"Error in P structural analysis: {e}")
        return False, 0.0

def analyze_r_structure(image):
    """Analyze if the image has R-like structure"""
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False, 0.0
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Calculate aspect ratio (width/height)
        aspect_ratio = w / h if h > 0 else 0
        
        # R typically has aspect ratio around 0.5-0.7
        aspect_ratio_score = 0.0
        if 0.4 < aspect_ratio < 0.8:
            aspect_ratio_score = 0.2
        
        # Divide image into top and bottom halves
        top_half = binary[y:y+int(h*0.5), x:x+w]
        bottom_half = binary[y+int(h*0.5):y+h, x:x+w]
        
        # Divide into left, middle, and right regions
        left_region = binary[y:y+h, x:x+int(w*0.3)]
        right_top_region = binary[y:y+int(h*0.5), x+int(w*0.3):x+w]
        right_bottom_region = binary[y+int(h*0.5):y+h, x+int(w*0.3):x+w]
        
        # Calculate white pixel ratios in each region
        left_ratio = np.sum(left_region > 0) / (left_region.shape[0] * left_region.shape[1]) if left_region.size > 0 else 0
        right_top_ratio = np.sum(right_top_region > 0) / (right_top_region.shape[0] * right_top_region.shape[1]) if right_top_region.size > 0 else 0
        right_bottom_ratio = np.sum(right_bottom_region > 0) / (right_bottom_region.shape[0] * right_bottom_region.shape[1]) if right_bottom_region.size > 0 else 0
        
        # R has strong vertical line on the left, curved region on top right, and diagonal line on bottom right
        region_score = 0.0
        if left_ratio > 0.6:  # Strong vertical line on the left
            region_score += 0.2
        
        if right_top_ratio > 0.3:  # Some density in top right for the curve
            region_score += 0.2
        
        # Check for diagonal line in bottom right
        # For a diagonal, we expect some white pixels in the bottom right, but not as many as top right
        if 0.1 < right_bottom_ratio < right_top_ratio:
            region_score += 0.2
        
        # Calculate diagonal detection
        has_diagonal = False
        bottom_right_quarter = binary[y+int(h*0.5):y+h, x+int(w*0.5):x+w]
        if bottom_right_quarter.size > 0:
            # Check if pixels are aligned in a diagonal pattern
            rows, cols = bottom_right_quarter.shape
            diagonal_sum = 0
            for i in range(min(rows, cols)):
                if i < rows and i < cols and bottom_right_quarter[i, i] > 0:
                    diagonal_sum += 1
            
            diagonal_ratio = diagonal_sum / min(rows, cols) if min(rows, cols) > 0 else 0
            if diagonal_ratio > 0.3:
                has_diagonal = True
        
        diagonal_score = 0.2 if has_diagonal else 0.0
        
        # Combine scores
        total_score = aspect_ratio_score + region_score + diagonal_score
        
        # Normalize to [0,1]
        normalized_score = min(1.0, total_score)
        
        is_r = normalized_score > 0.6
        
        print(f"R structural analysis: score={normalized_score:.2f}, is_r={is_r}")
        
        return is_r, normalized_score
    except Exception as e:
        print(f"Error in R structural analysis: {e}")
        return False, 0.0

def analyze_o_structure(image):
    """Analyze if the image has O-like structure"""
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False, 0.0
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Calculate aspect ratio (width/height)
        aspect_ratio = w / h if h > 0 else 0
        
        # O typically has aspect ratio close to 1
        aspect_ratio_score = 0.0
        if 0.7 < aspect_ratio < 1.3:
            aspect_ratio_score = 0.3
        
        # Calculate circularity
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # O has high circularity
        circularity_score = 0.0
        if circularity > 0.7:
            circularity_score = 0.4
        
        # Check for empty center
        mask = np.zeros_like(binary)
        cv2.drawContours(mask, [largest_contour], 0, 255, -1)
        
        # Create a smaller contour to check for hole
        center_x, center_y = x + w // 2, y + h // 2
        center_region = binary[center_y - h//4:center_y + h//4, center_x - w//4:center_x + w//4]
        
        # O should have lower density in the center
        center_density = np.sum(center_region > 0) / (center_region.shape[0] * center_region.shape[1]) if center_region.size > 0 else 1.0
        edge_density = (np.sum(binary > 0) - np.sum(center_region > 0)) / (binary.size - center_region.size) if (binary.size - center_region.size) > 0 else 0.0
        
        center_score = 0.0
        if center_density < edge_density * 0.5:  # Center should have less than half the density of the edges
            center_score = 0.3
        
        # Combine scores
        total_score = aspect_ratio_score + circularity_score + center_score
        
        # Normalize to [0,1]
        normalized_score = min(1.0, total_score)
        
        is_o = normalized_score > 0.6
        
        print(f"O structural analysis: score={normalized_score:.2f}, is_o={is_o}")
        
        return is_o, normalized_score
    except Exception as e:
        print(f"Error in O structural analysis: {e}")
        return False, 0.0

def analyze_q_structure(image):
    """Analyze if the image has Q-like structure"""
    # First check if it's O-like
    is_o, o_score = analyze_o_structure(image)
    
    if not is_o or o_score < 0.5:
        return False, 0.0
    
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False, 0.0
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Check bottom right quadrant for the diagonal tail
        bottom_right = binary[y+int(h*0.6):y+h, x+int(w*0.6):x+w]
        
        # Q should have some pixels in the bottom right for the tail
        tail_density = np.sum(bottom_right > 0) / (bottom_right.shape[0] * bottom_right.shape[1]) if bottom_right.size > 0 else 0
        
        tail_score = 0.0
        if tail_density > 0.2:
            tail_score = 0.3
        
        # Combine O score with tail score
        total_score = o_score * 0.7 + tail_score
        
        # Normalize to [0,1]
        normalized_score = min(1.0, total_score)
        
        is_q = normalized_score > 0.6
        
        print(f"Q structural analysis: score={normalized_score:.2f}, is_q={is_q}")
        
        return is_q, normalized_score
    except Exception as e:
        print(f"Error in Q structural analysis: {e}")
        return False, 0.0

def analyze_c_structure(image):
    """Analyze if the image has C-like structure"""
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False, 0.0
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Calculate aspect ratio (width/height)
        aspect_ratio = w / h if h > 0 else 0
        
        # C typically has aspect ratio around 0.7-1.0
        aspect_ratio_score = 0.0
        if 0.6 < aspect_ratio < 1.1:
            aspect_ratio_score = 0.2
        
        # Divide image into left and right regions
        left_region = binary[y:y+h, x:x+int(w*0.3)]
        middle_region = binary[y:y+h, x+int(w*0.3):x+int(w*0.7)]
        right_region = binary[y:y+h, x+int(w*0.7):x+w]
        
        # Calculate white pixel ratios in each region
        left_ratio = np.sum(left_region > 0) / (left_region.shape[0] * left_region.shape[1]) if left_region.size > 0 else 0
        middle_ratio = np.sum(middle_region > 0) / (middle_region.shape[0] * middle_region.shape[1]) if middle_region.size > 0 else 0
        right_ratio = np.sum(right_region > 0) / (right_region.shape[0] * right_region.shape[1]) if right_region.size > 0 else 0
        
        # C has higher density on left than right
        region_score = 0.0
        if left_ratio > right_ratio * 1.5:
            region_score += 0.3
        
        # Check for opening on the right
        top_right = binary[y:y+int(h*0.3), x+int(w*0.7):x+w]
        bottom_right = binary[y+int(h*0.7):y+h, x+int(w*0.7):x+w]
        middle_right = binary[y+int(h*0.3):y+int(h*0.7), x+int(w*0.7):x+w]
        
        # C should have some density in top right and bottom right, but less in middle right
        opening_score = 0.0
        if top_right.size > 0 and bottom_right.size > 0 and middle_right.size > 0:
            top_right_ratio = np.sum(top_right > 0) / top_right.size
            bottom_right_ratio = np.sum(bottom_right > 0) / bottom_right.size
            middle_right_ratio = np.sum(middle_right > 0) / middle_right.size
            
            if top_right_ratio > 0.2 and bottom_right_ratio > 0.2 and middle_right_ratio < 0.2:
                opening_score = 0.3
        
        # Combine scores
        total_score = aspect_ratio_score + region_score + opening_score
        
        # Normalize to [0,1]
        normalized_score = min(1.0, total_score)
        
        is_c = normalized_score > 0.6
        
        print(f"C structural analysis: score={normalized_score:.2f}, is_c={is_c}")
        
        return is_c, normalized_score
    except Exception as e:
        print(f"Error in C structural analysis: {e}")
        return False, 0.0

# Add functions for other letter structural analysis as needed

if __name__ == "__main__":
    # Simple test code
    if len(sys.argv) > 2:
        image_path = sys.argv[1]
        letter = sys.argv[2].upper()
        
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            sys.exit(1)
        
        # Load and process the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image: {image_path}")
            sys.exit(1)
        
        # Detect the specified letter
        is_letter, confidence = detect_specific_letter(image, letter)
        print(f"Detection result for letter {letter}: {is_letter} with confidence {confidence:.4f}")
    else:
        print("Usage: python specialized_letter_detector.py <image_path> <letter>")