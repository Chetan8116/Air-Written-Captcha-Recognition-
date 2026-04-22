"""
Specialized classifier utilities for improving 5 vs 3 distinction in air-written digits.
This module provides preprocessing and prediction functions for the specialized classifier.
"""

import numpy as np
import cv2
import tensorflow as tf
import os

def enhance_air_writing_features(image, enhance_level=1.5):
    """
    Enhance air-written digit features to improve recognition, especially for 5 vs 3.
    
    Args:
        image: Input image (grayscale, 28x28)
        enhance_level: Level of enhancement (default 1.5)
        
    Returns:
        Enhanced image with better feature definition
    """
    if len(image.shape) > 2 and image.shape[2] > 1:
        # Convert to grayscale if not already
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Ensure correct image size
    if image.shape != (28, 28):
        image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Make a copy to avoid modifying original
    enhanced = image.copy().astype(np.float32)
    
    # Apply adaptive thresholding to better separate foreground from background
    if np.max(enhanced) > 1.0:  # Check if image is already normalized
        # For non-normalized images
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
    else:
        # For normalized images
        temp = (enhanced * 255).astype(np.uint8)
        binary = cv2.adaptiveThreshold(
            temp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
    
    # Create a mask for critical 5 vs 3 distinguishing features
    # For digit 5: Enhance the top horizontal line and the curved bottom
    # For digit 3: Enhance the two curved segments
    
    # Sobel edges to detect key features
    sobelx = cv2.Sobel(binary, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(binary, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute gradient magnitude to identify strong edges
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Dilate important edges to make them more prominent
    kernel = np.ones((2, 2), np.uint8)
    dilated_edges = cv2.dilate(magnitude, kernel, iterations=1)
    
    # Blend enhanced edges with original image
    if np.max(enhanced) > 1.0:
        # For non-normalized images
        enhanced = cv2.addWeighted(
            enhanced, 0.7, 
            dilated_edges.astype(np.float32), 0.3 * enhance_level, 
            0
        )
    else:
        # For normalized images
        enhanced = cv2.addWeighted(
            enhanced, 0.7, 
            dilated_edges.astype(np.float32) / 255.0, 0.3 * enhance_level, 
            0
        )
    
    # Apply slight contrast enhancement
    if np.max(enhanced) > 1.0:
        # For non-normalized images
        enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    else:
        # For normalized images
        enhanced = cv2.normalize(enhanced, None, 0, 1, cv2.NORM_MINMAX)
    
    # Ensure output is in correct format (same as input)
    if np.max(image) <= 1.0 and np.max(enhanced) > 1.0:
        enhanced = enhanced / 255.0
    elif np.max(image) > 1.0 and np.max(enhanced) <= 1.0:
        enhanced = enhanced * 255.0
    
    return enhanced.astype(image.dtype)

def predict_with_specialized_model(image, model_path="5_vs_3_model.h5"):
    """
    Make a prediction using the specialized 5 vs 3 classifier.
    
    Args:
        image: Input image (grayscale, 28x28)
        model_path: Path to the specialized model
        
    Returns:
        Tuple of (predicted class (3 or 5), confidence score)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Specialized model not found at {model_path}")
    
    # Load model if not already loaded
    try:
        model = tf.keras.models.load_model(model_path)
    except:
        raise RuntimeError(f"Failed to load specialized model from {model_path}")
    
    # Preprocess image
    processed_image = enhance_air_writing_features(image)
    
    # Ensure correct shape for model input
    if len(processed_image.shape) < 4:
        # Add batch and channel dimensions
        if len(processed_image.shape) == 2:
            processed_image = processed_image.reshape(1, 28, 28, 1)
        elif len(processed_image.shape) == 3 and processed_image.shape[2] == 1:
            processed_image = processed_image.reshape(1, 28, 28, 1)
        else:
            raise ValueError(f"Unexpected image shape: {processed_image.shape}")
    
    # Ensure normalization
    if np.max(processed_image) > 1.0:
        processed_image = processed_image / 255.0
    
    # Make prediction
    prediction = model.predict(processed_image, verbose=0)
    
    # For binary classifier: output is likelihood of being digit 5 vs 3
    # If prediction[0][0] > 0.5, then it's more likely to be 5, otherwise 3
    confidence = float(abs(prediction[0][0] - 0.5) * 2)  # Convert to 0-1 scale
    predicted_class = 5 if prediction[0][0] > 0.5 else 3
    
    return predicted_class, confidence

def generate_sample_data():
    """
    Generate sample data for testing the specialized classifier.
    Creates simple artificial images of digits 3 and 5 for testing.
    
    Returns:
        Two 28x28 images - one representing a 3 and one representing a 5
    """
    # Create empty images
    digit_3 = np.zeros((28, 28), dtype=np.uint8)
    digit_5 = np.zeros((28, 28), dtype=np.uint8)
    
    # Draw a simple 3
    # Top curve
    cv2.ellipse(digit_3, (14, 7), (7, 5), 0, 180, 360, 255, -1)
    # Bottom curve
    cv2.ellipse(digit_3, (14, 21), (7, 5), 0, 180, 360, 255, -1)
    
    # Draw a simple 5
    # Top horizontal line
    cv2.line(digit_5, (7, 5), (21, 5), 255, 2)
    # Vertical line from top
    cv2.line(digit_5, (7, 5), (7, 14), 255, 2)
    # Middle horizontal line
    cv2.line(digit_5, (7, 14), (21, 14), 255, 2)
    # Bottom curve
    cv2.ellipse(digit_5, (14, 21), (7, 5), 0, 180, 360, 255, -1)
    
    return digit_3, digit_5

def test_specialized_classifier(model_path="5_vs_3_model.h5"):
    """
    Test the specialized classifier with sample data.
    
    Args:
        model_path: Path to the specialized model
        
    Returns:
        Boolean indicating if the test was successful
    """
    if not os.path.exists(model_path):
        print(f"Error: Specialized model not found at {model_path}")
        return False
    
    # Generate sample data
    digit_3, digit_5 = generate_sample_data()
    
    # Save sample images for inspection
    cv2.imwrite("sample_3.png", digit_3)
    cv2.imwrite("sample_5.png", digit_5)
    
    # Test predictions
    try:
        class_3, conf_3 = predict_with_specialized_model(digit_3, model_path)
        class_5, conf_5 = predict_with_specialized_model(digit_5, model_path)
        
        print(f"Sample 3 predicted as: {class_3} with confidence: {conf_3:.2f}")
        print(f"Sample 5 predicted as: {class_5} with confidence: {conf_5:.2f}")
        
        # Check if predictions are correct
        if class_3 == 3 and class_5 == 5:
            print("Specialized classifier is working correctly!")
            return True
        else:
            print("Specialized classifier made incorrect predictions.")
            return False
    except Exception as e:
        print(f"Error testing specialized classifier: {e}")
        return False

if __name__ == "__main__":
    print("Testing specialized classifier utilities...")
    
    # Generate sample data for testing
    digit_3, digit_5 = generate_sample_data()
    
    # Save sample images for inspection
    cv2.imwrite("sample_3.png", digit_3)
    cv2.imwrite("sample_5.png", digit_5)
    
    # Test enhancement function
    enhanced_3 = enhance_air_writing_features(digit_3)
    enhanced_5 = enhance_air_writing_features(digit_5)
    
    cv2.imwrite("enhanced_3.png", enhanced_3)
    cv2.imwrite("enhanced_5.png", enhanced_5)
    
    print("Sample images and enhanced versions saved for inspection.")
    
    # Test classifier if available
    if os.path.exists("5_vs_3_model.h5"):
        test_specialized_classifier()
    else:
        print("Specialized model not found. Run train_5_vs_3_classifier.py first.")