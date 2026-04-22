"""
B MNIST Detector Integration
This module provides functions to use the specialized B MNIST-style detector
"""

import os
import numpy as np
import cv2
import tensorflow as tf

# Global variables for model
b_mnist_model = None
model_loaded = False

def load_b_mnist_model():
    """Load the B MNIST-style detector model"""
    global b_mnist_model, model_loaded
    
    try:
        # Define model path
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "specialized")
        model_path = os.path.join(model_dir, "b_mnist_detector.h5")
        
        # Alternative paths
        alt_paths = [
            os.path.join(model_dir, "b_detector.h5"),  # Try existing detector
        ]
        
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"MNIST B detector not found at {model_path}")
            # Try alternative paths
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    model_path = alt_path
                    print(f"Using alternative model: {model_path}")
                    break
            else:
                print("No B detector model found")
                return False
        
        # Load model
        b_mnist_model = tf.keras.models.load_model(model_path)
        model_loaded = True
        print(f"B MNIST detector model loaded successfully from {model_path}")
        return True
    
    except Exception as e:
        print(f"Error loading B MNIST detector model: {e}")
        return False

def preprocess_for_b_mnist(image):
    """Preprocess image for B MNIST detector"""
    try:
        # Convert to grayscale if needed
        if len(image.shape) > 2 and image.shape[2] > 1:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Make sure image is uint8
        if np.max(gray) <= 1.0:
            gray = (gray * 255).astype(np.uint8)
        
        # Apply thresholding to get binary image
        _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours to crop to content
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get bounding box of largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add padding
            padding = int(max(w, h) * 0.1)  # 10% padding
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(binary.shape[1] - x, w + 2 * padding)
            h = min(binary.shape[0] - y, h + 2 * padding)
            
            # Crop to content
            cropped = binary[y:y+h, x:x+w]
            
            # Make square by adding padding
            size = max(cropped.shape[0], cropped.shape[1])
            square = np.zeros((size, size), dtype=np.uint8)
            
            # Center the content
            offset_x = (size - cropped.shape[1]) // 2
            offset_y = (size - cropped.shape[0]) // 2
            square[offset_y:offset_y+cropped.shape[0], offset_x:offset_x+cropped.shape[1]] = cropped
            
            # Resize to 28x28
            resized = cv2.resize(square, (28, 28))
        else:
            # If no contours, just resize the binary image
            resized = cv2.resize(binary, (28, 28))
        
        # Save processed image for debugging
        debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_output")
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "b_mnist_processed.png"), resized)
        
        # Normalize and reshape for model
        normalized = resized / 255.0
        mnist_input = normalized.reshape(1, 28, 28, 1)
        
        return mnist_input
    
    except Exception as e:
        print(f"Error preprocessing image for B MNIST detector: {e}")
        return None

def is_letter_b_mnist(image, threshold=0.5):
    """
    Check if the image contains the letter B using MNIST-style model
    
    Args:
        image: Input image
        threshold: Confidence threshold (default: 0.5)
        
    Returns:
        tuple: (is_b, confidence) - Boolean indicating if it's a B and confidence score
    """
    global b_mnist_model, model_loaded
    
    try:
        # Load model if not already loaded
        if not model_loaded:
            if not load_b_mnist_model():
                print("Could not load B MNIST detector model")
                return False, 0.0
        
        # Preprocess image
        mnist_input = preprocess_for_b_mnist(image)
        
        if mnist_input is None:
            print("Failed to preprocess image for B MNIST detector")
            return False, 0.0
        
        # Make prediction
        prediction = b_mnist_model.predict(mnist_input, verbose=0)[0]
        
        # Get confidence for B (class 1)
        b_confidence = float(prediction[1])
        
        print(f"B MNIST detector prediction: {prediction}, B confidence: {b_confidence:.4f}")
        
        # Return result
        is_b = b_confidence > threshold
        return is_b, b_confidence
    
    except Exception as e:
        print(f"Error in B MNIST detection: {e}")
        return False, 0.0

# Initialize model at module load time
try:
    load_b_mnist_model()
except Exception as e:
    print(f"Error initializing B MNIST detector module: {e}")