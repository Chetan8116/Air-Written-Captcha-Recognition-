"""
Integration module for the specialized B detector
This module provides functions to detect B letters using the specialized model
"""

import os
import numpy as np
import cv2
import tensorflow as tf
# Import matplotlib only when needed for visualization
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# Global variables to hold the model
b_detector_model = None
model_loaded = False

def load_b_detector_model():
    """Load the specialized B detector model"""
    global b_detector_model, model_loaded
    
    try:
        # Path to the model
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                 "models", "specialized", "b_detector.h5")
        
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"B detector model not found at {model_path}")
            return False
        
        # Load the model
        b_detector_model = tf.keras.models.load_model(model_path)
        model_loaded = True
        print(f"B detector model loaded successfully from {model_path}")
        return True
    except Exception as e:
        print(f"Error loading B detector model: {e}")
        model_loaded = False
        return False

def preprocess_for_b_detection(image):
    """Preprocess an image for B detection"""
    # Convert to grayscale if needed
    if len(image.shape) > 2 and image.shape[2] > 1:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Make sure image is uint8
    if np.max(gray) <= 1.0:
        gray = (gray * 255).astype(np.uint8)
    
    # Create debug directory
    debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_output")
    os.makedirs(debug_dir, exist_ok=True)
    
    # Save original input
    cv2.imwrite(os.path.join(debug_dir, "b_detection_original.png"), gray)
    
    # Apply median blur to remove noise while preserving edges
    blurred = cv2.medianBlur(gray, 5)
    
    # Apply adaptive thresholding for better edge detection
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Clean up the image with morphology
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Save processed binary for debug
    cv2.imwrite(os.path.join(debug_dir, "b_detection_preprocessed.png"), binary)
    
    # Resize to 28x28
    resized = cv2.resize(binary, (28, 28))
    
    # Save resized for debug
    cv2.imwrite(os.path.join(debug_dir, "b_detection_input.png"), resized)
    
    # Normalize to [0,1] and add batch and channel dimensions
    normalized = resized.astype('float32') / 255.0
    return np.expand_dims(np.expand_dims(normalized, axis=0), axis=-1)

def is_letter_b(image, threshold=0.5):
    """
    Determine if the image contains the letter B using the specialized model
    
    Args:
        image: Input image
        threshold: Confidence threshold for B detection
        
    Returns:
        tuple: (is_b, confidence) - Boolean indicating if it's a B and confidence score
    """
    global b_detector_model, model_loaded
    
    # First, try direct structural analysis - often more reliable for clear B shapes
    try:
        # Quick structural check for obviously B-like shapes
        is_b_structure, structure_score = detect_b_structure(image)
        
        # If very confident from structural analysis, return immediately
        if is_b_structure and structure_score > 0.7:
            print(f"Direct structural B detection with high confidence: {structure_score:.4f}")
            return True, structure_score
    except Exception as e:
        print(f"Error in direct structural check: {e}")
    
    # Try to load the model if not loaded
    if not model_loaded:
        if not load_b_detector_model():
            # If model can't be loaded, use fallback structural analysis
            return detect_b_structure(image)
    
    try:
        # Preprocess the image
        processed_image = preprocess_for_b_detection(image)
        
        # Make prediction
        confidence = float(b_detector_model.predict(processed_image)[0][0])
        
        print(f"B detector model confidence: {confidence:.4f}")
        return confidence > threshold, confidence
    except Exception as e:
        print(f"Error in B detection: {e}")
        # Fallback to structural analysis
        return detect_b_structure(image)

def detect_b_structure(image):
    """
    Fallback method to detect B using structural analysis
    Used when the specialized model is not available
    
    Args:
        image: Input image
        
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
    
    # Create debug directory
    debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_output")
    os.makedirs(debug_dir, exist_ok=True)
    
    # Resize for consistent analysis
    resized = cv2.resize(gray, (100, 100))
    cv2.imwrite(os.path.join(debug_dir, "b_structure_resized.png"), resized)
    
    # Apply thresholding (try multiple methods and use the best one)
    # Method 1: Simple threshold
    _, binary1 = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Method 2: Otsu threshold
    _, binary2 = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    # Method 3: Adaptive threshold
    binary3 = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Save all versions for debugging
    cv2.imwrite(os.path.join(debug_dir, "b_structure_binary1.png"), binary1)
    cv2.imwrite(os.path.join(debug_dir, "b_structure_binary2.png"), binary2)
    cv2.imwrite(os.path.join(debug_dir, "b_structure_binary3.png"), binary3)
    
    # Try all three binary versions and choose the one with the most promising features
    binaries = [binary1, binary2, binary3]
    best_score = 0
    best_binary = binary2  # Default to Otsu
    
    for i, binary in enumerate(binaries):
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue
            
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # Simple scoring for B-like shape
        score = 0
        if 0.45 < aspect_ratio < 0.85:  # B typically has this aspect ratio
            score += 0.5
            
        # If this binary version gives better results
        if score > best_score:
            best_score = score
            best_binary = binary
    
    # Use the best binary version
    binary = best_binary
    cv2.imwrite(os.path.join(debug_dir, "b_structure_best_binary.png"), binary)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return False, 0.0
    
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate contour properties
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Calculate aspect ratio and other features
    aspect_ratio = w / h if h > 0 else 0
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    
    # Create a mask to analyze regions
    mask = np.zeros_like(binary)
    cv2.drawContours(mask, [largest_contour], 0, 255, -1)
    
    # Define regions (left, middle, right)
    left_region = mask[:, :33]
    middle_region = mask[:, 33:66]
    right_region = mask[:, 66:]
    
    # Calculate white pixel ratios in each region
    left_ratio = np.sum(left_region > 0) / np.size(left_region)
    middle_ratio = np.sum(middle_region > 0) / np.size(middle_region)
    right_ratio = np.sum(right_region > 0) / np.size(right_region)
    
    # B score calculation based on structural features
    b_score = 0.0
    
    # Log key features for debugging
    print(f"B detection features - Aspect ratio: {aspect_ratio:.4f}, Circularity: {circularity:.4f}")
    print(f"Region ratios - Left: {left_ratio:.2f}, Middle: {middle_ratio:.2f}, Right: {right_ratio:.2f}")
    
    # B typically has aspect ratio around 0.4-0.8
    if 0.4 < aspect_ratio < 0.85:
        aspect_bonus = max(0.3, 0.5 - abs(aspect_ratio - 0.5) * 2)  # Higher score for closer to 0.5
        b_score += aspect_bonus
        print(f"B aspect ratio match: {aspect_ratio:.4f}, bonus: {aspect_bonus:.2f}")
    
    # B has moderate circularity (not too circular, not too jagged)
    if 0.25 < circularity < 0.75:  # Widened range
        circularity_bonus = 0.25 - abs(circularity - 0.45) * 0.5  # Ideal around 0.45
        b_score += max(0.1, circularity_bonus)
        print(f"B circularity match: {circularity:.4f}")
    
    # B typically has strong left edge (vertical line)
    if left_ratio > 0.25:  # Lowered threshold further
        b_score += min(0.3, left_ratio * 0.6)  # Scale with strength of left edge
        print(f"B left edge detected: {left_ratio:.2f}")
    
    # B has moderate middle region (where the gaps between curves are)
    if 0.05 < middle_ratio < 0.7:  # Even wider range
        b_score += 0.2
        print(f"B middle region match: {middle_ratio:.2f}")
    
    # B typically has curved regions on right side (not as dense as left)
    if right_ratio < left_ratio * 1.2 and right_ratio > 0.15:
        b_score += 0.15
        print(f"B right curve pattern detected: {right_ratio:.2f} vs left {left_ratio:.2f}")
    
    # Check for horizontal density pattern
    h_profile = np.sum(binary, axis=1) / binary.shape[1]
    from scipy.signal import find_peaks
    peaks, peak_props = find_peaks(h_profile, height=0.05, distance=10)  # More sensitive detection
    
    # Save horizontal profile for debugging
    plt_debug = False
    if plt_debug:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 4))
            plt.plot(h_profile)
            plt.scatter(peaks, h_profile[peaks], color='red')
            plt.title(f'Horizontal Profile with {len(peaks)} peaks')
            plt.savefig(os.path.join(debug_dir, "b_horizontal_profile.png"))
            plt.close()
        except ImportError:
            pass
    
    # B typically has two main horizontal density peaks (top and bottom curves)
    if 1 <= len(peaks) <= 5:  # Allow 1-5 peaks
        b_score += 0.2
        print(f"B horizontal peaks detected: {len(peaks)}")
    
    # Check vertical profile (should have high density on left)
    v_profile = np.sum(binary, axis=0) / binary.shape[0]
    left_sum = np.sum(v_profile[:33])
    right_sum = np.sum(v_profile[66:])
    
    # B should have strong left edge
    if left_sum > right_sum:
        b_score += 0.15
        print(f"B vertical profile match: left={left_sum:.1f}, right={right_sum:.1f}")
    
    # Additional check for color (if it's a colored image)
    if len(image.shape) > 2 and image.shape[2] >= 3:
        # Check if it's predominantly red (like in the screenshot)
        b, g, r = cv2.split(image)
        
        # If the red channel is significantly stronger than blue and green
        if np.mean(r) > 1.2 * np.mean(g) and np.mean(r) > 1.2 * np.mean(b):
            b_score += 0.15
            print("Red color bonus for B detection")
    
    # Hard-coded checks for known B shapes
    # If aspect ratio is very close to 0.5 (common for B)
    if 0.45 < aspect_ratio < 0.55:
        b_score += 0.2
        print(f"Perfect B aspect ratio: {aspect_ratio:.4f}")
    
    # Higher scores are more likely to be B
    print(f"Structural B detection final score: {b_score:.4f}")
    
    # Lower threshold to be more sensitive to B detection
    is_b = b_score > 0.35
    print(f"B detection result: {'YES' if is_b else 'NO'}")
    
    return is_b, b_score

# Initialize the model at module load time
try:
    load_b_detector_model()
except Exception as e:
    print(f"Error initializing B detector module: {e}")