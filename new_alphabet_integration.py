"""
Integration module for the Alphabet CNN model from the alphabets/models directory
This module provides seamless integration of the model with the VirtualPainter app
"""

import os
import numpy as np
import tensorflow as tf
import cv2
import json
from typing import Tuple, Dict, Optional, Union, List
from specialized_letter_detection import detect_specific_letter
# Import MNIST-style B detector
try:
    from b_mnist_detector import is_letter_b_mnist
except ImportError:
    print("Warning: b_mnist_detector module not found")
    is_letter_b_mnist = None

# Global variables for model storage
alphabet_model = None
alphabet_class_mapping = {}
model_loaded = False

def load_alphabet_model() -> bool:
    """
    Load the alphabet model from the alphabets/models directory
    
    Returns:
        bool: True if loading was successful, False otherwise
    """
    global alphabet_model, alphabet_class_mapping, model_loaded
    
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # First try the converted model that's known to be compatible
    model_path = os.path.join(base_dir, "models", "compatible", "alphabet_model_converted.h5")
    # Fallback to the original model path if the converted one doesn't exist
    if not os.path.exists(model_path):
        model_path = os.path.join(base_dir, "alphabets", "models", "alphabet_model_best.h5")
    
    # Try finding the mapping file in multiple locations
    mapping_path = os.path.join(base_dir, "models", "compatible", "alphabet_class_mapping.json")
    if not os.path.exists(mapping_path):
        mapping_path = os.path.join(base_dir, "alphabets", "models", "alphabet_class_mapping.json")
    
    try:
        print(f"Loading alphabet CNN model from {model_path}...")
        
        # Check if the model file exists
        if not os.path.exists(model_path):
            print(f"Error: Model file {model_path} does not exist")
            return False
            
        # Check if the mapping file exists
        if not os.path.exists(mapping_path):
            print(f"Error: Mapping file {mapping_path} does not exist")
            return False
        
        # Load the class mapping
        with open(mapping_path, 'r') as f:
            alphabet_class_mapping = json.load(f)
        print(f"Class mapping loaded with {len(alphabet_class_mapping)} classes")
        
        # Try to load the model with a compatibility fix for batch_shape issue
        try:
            # First attempt: normal loading
            alphabet_model = tf.keras.models.load_model(model_path, compile=False)
        except Exception as load_error:
            print(f"Standard model loading failed: {load_error}")
            
            # Second attempt: Try loading with custom objects
            try:
                # Load model with custom object scope to handle InputLayer with batch_shape
                custom_objects = {
                    'InputLayer': lambda config: tf.keras.layers.InputLayer(
                        input_shape=config.get('batch_shape')[1:] if config.get('batch_shape') else None,
                        sparse=config.get('sparse', False),
                        ragged=config.get('ragged', False),
                        name=config.get('name')
                    )
                }
                with tf.keras.utils.custom_object_scope(custom_objects):
                    alphabet_model = tf.keras.models.load_model(model_path, compile=False)
                print("Model loaded successfully using custom object scope")
            except Exception as custom_load_error:
                print(f"Custom object loading failed: {custom_load_error}")
                
                # Third attempt: Try to use model conversion logic
                try:
                    from convert_alphabet_model import convert_model
                    
                    # Convert the model to a temporary file
                    tmp_model_path = os.path.join(base_dir, "models", "compatible", "temp_converted_model.h5")
                    os.makedirs(os.path.dirname(tmp_model_path), exist_ok=True)
                    
                    print(f"Attempting to convert model from {model_path} to {tmp_model_path}...")
                    if convert_model(model_path, tmp_model_path):
                        # Try to load the converted model
                        alphabet_model = tf.keras.models.load_model(tmp_model_path, compile=False)
                        print(f"Successfully loaded converted model from {tmp_model_path}")
                    else:
                        raise Exception("Model conversion failed")
                except Exception as conversion_error:
                    print(f"Model conversion failed: {conversion_error}")
                    raise Exception(f"All model loading attempts failed. Original error: {load_error}")
        
        # Compile the model if needed
        if not getattr(alphabet_model, 'compiled_loss', None):
            alphabet_model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
        model_loaded = True
        print("Alphabet model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading alphabet model: {e}")
        model_loaded = False
        return False

def is_model_loaded() -> bool:
    """
    Check if the alphabet model is loaded
    
    Returns:
        bool: True if model is loaded, False otherwise
    """
    global model_loaded
    return model_loaded

def preprocess_image(image: np.ndarray, enhance_uppercase=True) -> np.ndarray:
    """
    Preprocess the image for the alphabet model with enhanced processing for uppercase letters
    
    Args:
        image: Input image (grayscale, 28x28 or can be resized)
        enhance_uppercase: Apply additional enhancement for uppercase recognition
        
    Returns:
        np.ndarray: Preprocessed image ready for model input
    """
    # Convert to grayscale if it's not already
    if len(image.shape) > 2 and image.shape[2] > 1:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Convert to uint8 if needed
    if np.max(image) <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Resize to a larger dimension for better preprocessing and then down to 28x28
    if image.shape[0] != 28 or image.shape[1] != 28:
        # First resize to larger dimension for better processing
        large_size = 100  # Processing at a larger size helps retain details
        image_large = cv2.resize(image, (large_size, large_size), interpolation=cv2.INTER_CUBIC)
    else:
        # Upscale for better processing
        image_large = cv2.resize(image, (100, 100), interpolation=cv2.INTER_CUBIC)
    
    if enhance_uppercase:
        # Enhanced preprocessing pipeline focused on uppercase letter recognition
        
        # Step 1: Noise reduction with bilateral filter (preserves edges better than Gaussian)
        denoised = cv2.bilateralFilter(image_large, 9, 75, 75)
        
        # Step 2: Convert to binary using Otsu's method (more robust than adaptive thresholding for clear strokes)
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Step 3: Morphological operations to improve letter shapes
        # Create a kernel that's more suited for letter structure
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Close operation to connect nearby parts (helps with broken strokes)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Dilate to thicken lines slightly (helps with thin strokes)
        dilated = cv2.dilate(closed, kernel, iterations=1)
        
        # Step 4: Resize back to 28x28 with better interpolation for downsampling
        processed = cv2.resize(dilated, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Step 5: Ensure proper normalization
        processed = processed.astype(np.float32) / 255.0
        
    else:
        # Standard preprocessing for general case
        # Resize to model input size
        processed = cv2.resize(image_large, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Basic contrast enhancement
        processed = cv2.equalizeHist(processed)
        
        # Normalize
        processed = processed.astype(np.float32) / 255.0
    
    # Add extra processing for B, D, P, and R recognition which can be confused
    # This uses structural analysis to help distinguish these letters
    if enhance_uppercase:
        # Create a copy of processed image as uint8
        processed_uint8 = (processed * 255).astype(np.uint8)
        
        # Find contours to analyze shape
        contours, _ = cv2.findContours(processed_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Create a mask of the contour
            mask = np.zeros_like(processed_uint8)
            cv2.drawContours(mask, [largest_contour], -1, 255, -1)
            
            # Analyze the vertical symmetry (helps identify B, D, P, R)
            h, w = mask.shape
            left_half = mask[:, :w//2]
            right_half = mask[:, w//2:]
            
            # If the right half has significantly fewer white pixels, likely B or P
            # This helps strengthen the features specific to B and P
            right_pixels = np.sum(right_half > 0)
            left_pixels = np.sum(left_half > 0)
            
            if left_pixels > 0 and right_pixels / left_pixels < 0.7:
                # Enhance the right edge features to make B more distinctive
                edge_kernel = np.ones((2, 2), np.uint8)
                edges = cv2.Canny(mask, 100, 200)
                edges = cv2.dilate(edges, edge_kernel, iterations=1)
                
                # Combine with the processed image to enhance B-like features
                processed = np.maximum(processed, edges.astype(np.float32) / 255.0 * 0.5)
    
    # Reshape for model input (batch_size, height, width, channels)
    return processed.reshape(1, 28, 28, 1)

def predict_alphabet(image: np.ndarray, uppercase_only=False) -> Tuple[str, float, Dict]:
    """
    Predict the alphabet character from the input image with improved accuracy for uppercase
    
    Args:
        image: Input image (can be in any format, will be preprocessed)
        uppercase_only: Whether to force uppercase prediction only
        
    Returns:
        Tuple[str, float, Dict]: Predicted character, confidence, and all results
    """
    global alphabet_model, alphabet_class_mapping, model_loaded
    
    if not model_loaded or alphabet_model is None:
        if not load_alphabet_model():
            return "", 0.0, {}
    
    try:
        # Enhanced preprocessing for uppercase characters
        processed_image = preprocess_image(image, enhance_uppercase=True)
        
        # Make prediction
        predictions = alphabet_model.predict(processed_image, verbose=0)[0]
        
        # Get top 5 predictions for better decision making (increased from 3)
        top_indices = np.argsort(predictions)[-5:][::-1]
        top_chars = [alphabet_class_mapping.get(str(idx), "") for idx in top_indices]
        top_confidences = [float(predictions[idx]) for idx in top_indices]
        
        # Get the predicted class index and confidence
        predicted_idx = top_indices[0]
        confidence = top_confidences[0]
        
        # Get the predicted character from the mapping
        predicted_char = top_chars[0]
        
        # Special handling for uppercase B recognition which can be problematic
        # Check for structural features typical of B in the input image
        if uppercase_only:
            # Convert to uint8 for contour analysis
            if len(image.shape) > 2 and image.shape[2] > 1:
                img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = image.copy()
            
            # Ensure image is in right format
            if np.max(img_gray) <= 1.0:
                img_gray = (img_gray * 255).astype(np.uint8)
                
            # Resize for analysis if needed
            if img_gray.shape[0] != 100 or img_gray.shape[1] != 100:
                img_gray = cv2.resize(img_gray, (100, 100))
            
            # Apply threshold
            _, binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Calculate contour properties
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                
                # Bounding rectangle
                x, y, w, h = cv2.boundingRect(largest_contour)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                # Circularity (1 for a perfect circle)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                # Create a mask to analyze left and right portions
                mask = np.zeros_like(binary)
                cv2.drawContours(mask, [largest_contour], 0, 255, -1)
                
                # Divide the contour mask into left and right halves
                left_half = mask[:, :50]
                right_half = mask[:, 50:]
                
                # Check the ratio of white pixels in left vs right
                left_white = np.sum(left_half > 0)
                right_white = np.sum(right_half > 0)
                left_right_ratio = left_white / right_white if right_white > 0 else 999
                
                # Detect horizontal density patterns typical of B
                # B typically has a strong vertical line on the left and two curved segments on the right
                
                # Use our specialized letter detection module
                try:
                    # Check if it's a B using specialized detection
                    is_b, b_score = detect_specific_letter(img_gray, 'B')
                    
                    # Also try MNIST-style B detector if available
                    is_b_mnist = False
                    mnist_score = 0.0
                    
                    if is_letter_b_mnist is not None:
                        try:
                            is_b_mnist, mnist_score = is_letter_b_mnist(image)
                            print(f"MNIST B detector in alphabet integration: is_b={is_b_mnist}, score={mnist_score:.2f}")
                        except Exception as mnist_error:
                            print(f"Error in MNIST B detection: {mnist_error}")
                    
                    # Combine evidence from both detectors
                    b_detected = is_b or is_b_mnist
                    combined_score = max(b_score, mnist_score * 1.2)  # Give MNIST a 20% boost
                    
                    if b_detected:
                        print(f"Specialized B detection in alphabet integration: score={combined_score:.2f}")
                        
                        # Check if B is in the top predictions
                        b_idx = -1
                        for i, char in enumerate(top_chars):
                            if char == 'B':
                                b_idx = i
                                break
                        
                        # If B is in the top predictions, boost its confidence based on our detection
                        if b_idx >= 0:
                            # Significantly boost B's confidence
                            boosted_confidence = top_confidences[b_idx] * (1.0 + combined_score * 0.6)
                            if boosted_confidence > confidence:  # If the boost makes B the winner
                                predicted_char = 'B'
                                confidence = min(boosted_confidence, 0.95)  # Cap at 0.95
                        # If B was detected but isn't in the top predictions
                        # MNIST gets a lower threshold due to its higher accuracy
                        elif (b_score > 0.7) or (is_b_mnist and mnist_score > 0.6):
                            predicted_char = 'B'
                            confidence = combined_score * 0.75  # Higher confidence with combined approach
                except Exception as detector_error:
                    print(f"Error in specialized B detection: {detector_error}")
                    
                    # Fallback to traditional method if specialized detection fails
                    b_like_score = 0
                    
                    # B typically has aspect ratio around 0.5-0.7 (width/height)
                    if 0.45 < aspect_ratio < 0.8:
                        b_like_score += 1
                    
                    # B typically has strong left side
                    if left_right_ratio > 1.2:
                        b_like_score += 1
                    
                    # B has moderate circularity (not as circular as O, not as angular as E)
                    if 0.3 < circularity < 0.7:
                        b_like_score += 1
                        
                    # Check if B or similar letters (D, P, R) are in the top predictions
                    b_related_chars = ['B', 'D', 'P', 'R']
                    b_in_top = any(char in b_related_chars for char in top_chars)
                    
                    # If the shape appears to be B-like and B or similar is in the top predictions
                    if b_like_score >= 2 and b_in_top:
                        # Boost B in the predictions if it's present or reasonably likely
                        b_idx = -1
                        for i, char in enumerate(top_chars):
                            if char == 'B':
                                b_idx = i
                                break
                        
                        # If B is in the top predictions, boost its confidence
                        if b_idx >= 0:
                            # Significantly boost B's confidence
                            boosted_confidence = top_confidences[b_idx] * 1.5
                            if boosted_confidence > confidence:  # If the boost makes B the winner
                                predicted_char = 'B'
                                confidence = min(boosted_confidence, 0.95)  # Cap at 0.95
                        elif 'D' in top_chars or 'P' in top_chars or 'R' in top_chars:
                            # If these similar letters are present but B isn't, check if B might be a better match
                            if b_like_score >= 2.5:  # Higher threshold for replacing other letters with B
                                predicted_char = 'B'
                                confidence = 0.65  # Moderate confidence for this override
        
        # General uppercase handling
        if uppercase_only:
            # Look for the highest confidence uppercase letter if not already uppercase
            if not predicted_char.isupper():
                uppercase_idx = -1
                uppercase_confidence = 0
                
                for idx, char in enumerate(top_chars):
                    if char.isupper() and top_confidences[idx] > uppercase_confidence:
                        uppercase_idx = idx
                        uppercase_confidence = top_confidences[idx]
                
                # If we found an uppercase letter in the top predictions with decent confidence
                if uppercase_idx >= 0 and uppercase_confidence > 0.2:
                    predicted_char = top_chars[uppercase_idx]
                    confidence = uppercase_confidence
                else:
                    # Force uppercase version of the predicted char
                    predicted_char = predicted_char.upper()
        
        # Create detailed results dictionary for debugging
        results = {
            "predictions": {alphabet_class_mapping.get(str(i), f"Class_{i}"): float(pred) 
                           for i, pred in enumerate(predictions)},
            "predicted_index": int(predicted_idx),
            "confidence": confidence,
            "top_chars": top_chars,
            "top_confidences": top_confidences
        }
        
        return predicted_char, confidence, results
    except Exception as e:
        print(f"Error during alphabet prediction: {e}")
        return "", 0.0, {"error": str(e)}

# Initialize by trying to load the model immediately
try:
    load_alphabet_model()
except Exception as e:
    print(f"Error during initialization: {e}")
    model_loaded = False