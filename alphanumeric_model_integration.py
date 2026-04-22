"""
Integration script for the trained alphanumeric CAPTCHA model
This module provides functions to load and use the trained model in the VirtualPainter system
"""

import cv2
import numpy as np
import tensorflow as tf
import json
import os
from typing import Optional, Tuple, Dict, Any

class AlphanumericModelLoader:
    """Class to load and manage the trained alphanumeric model"""
    
    def __init__(self):
        self.model = None
        self.labels = {}
        self.label_mapping = {}
        self.is_loaded = False
        
    def load_model(self, model_path: str = "alphanumeric_model.h5", 
                   labels_path: str = "alphanumeric_labels.json") -> bool:
        """
        Load the trained alphanumeric model and label mappings
        
        Args:
            model_path: Path to the saved model file
            labels_path: Path to the label mappings JSON file
            
        Returns:
            bool: True if successfully loaded, False otherwise
        """
        try:
            # Check if files exist
            if not os.path.exists(model_path):
                print(f"Error: Model file '{model_path}' not found.")
                return False
                
            if not os.path.exists(labels_path):
                print(f"Error: Labels file '{labels_path}' not found.")
                return False
            
            # Load the model
            print(f"Loading alphanumeric model from {model_path}...")
            self.model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully!")
            
            # Load label mappings
            with open(labels_path, 'r') as f:
                self.labels = json.load(f)
            
            # Create reverse mapping (index to character)
            if 'classes' in self.labels:
                self.label_mapping = {i: char for i, char in enumerate(self.labels['classes'])}
                print(f"Label mappings loaded: {len(self.label_mapping)} classes")
                print(f"Available characters: {sorted(self.labels['classes'])}")
            else:
                print("Warning: No 'classes' found in labels file")
                return False
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model prediction
        
        Args:
            image: Input image (grayscale or color)
            
        Returns:
            np.ndarray: Preprocessed image ready for prediction
        """
        if image is None:
            return None
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize to model input size (28x28)
        resized = cv2.resize(gray, (28, 28))
        
        # Normalize pixel values
        normalized = resized.astype('float32') / 255.0
        
        # Reshape for model input (add batch and channel dimensions)
        input_image = normalized.reshape(1, 28, 28, 1)
        
        return input_image
    
    def predict_character(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Predict character from image
        
        Args:
            image: Input image
            
        Returns:
            Tuple[str, float]: (predicted_character, confidence)
        """
        if not self.is_loaded:
            return "", 0.0
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            if processed_image is None:
                return "", 0.0
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            
            # Get predicted class and confidence
            predicted_class = int(np.argmax(predictions, axis=1)[0])
            confidence = float(np.max(predictions))
            
            # Map to character
            if predicted_class not in self.label_mapping:
                return "", 0.0
                
            predicted_char = self.label_mapping[predicted_class]
            
            # Apply specialized digit classifiers for common confusions
            # Check if the character is a digit that might benefit from specialized classification
            try:
                # Import specialized classifier utilities if needed
                DIGIT_CLASSIFIER_AVAILABLE = False
                try:
                    from digit_classifier_utils import (
                        is_potentially_digit_eight, 
                        enhance_8_vs_3_features,
                        distinguish_6_vs_9,
                        distinguish_0_vs_6,
                        distinguish_1_vs_7,
                        enhance_digit_features
                    )
                    DIGIT_CLASSIFIER_AVAILABLE = True
                except ImportError:
                    DIGIT_CLASSIFIER_AVAILABLE = False
                
                if DIGIT_CLASSIFIER_AVAILABLE and predicted_char in ['0', '1', '3', '6', '7', '8', '9']:
                    # Extract the 28x28 image for analysis from the processed image
                    digit_img = processed_image.reshape(28, 28, 1)[..., 0]
                    
                    # Case 1: If predicted as '3', check if it might actually be an '8'
                    if predicted_char == '3':
                        if is_potentially_digit_eight(digit_img):
                            # Try with enhanced features
                            enhanced_features = enhance_8_vs_3_features(digit_img)
                            enhanced_input = enhanced_features.reshape(1, 28, 28, 1)
                            
                            # Get new prediction with enhanced features
                            new_predictions = self.model.predict(enhanced_input, verbose=0)[0]
                            
                            # Find indices for digits '3' and '8' in the label mapping
                            idx_3 = next((i for i, v in self.label_mapping.items() if v == '3'), None)
                            idx_8 = next((i for i, v in self.label_mapping.items() if v == '8'), None)
                            
                            if idx_3 is not None and idx_8 is not None:
                                confidence_3 = new_predictions[idx_3]  # Confidence for digit 3
                                confidence_8 = new_predictions[idx_8]  # Confidence for digit 8
                                
                                # If confidence for 8 is reasonable, change prediction
                                if confidence_8 > 0.2 and confidence_8 > confidence_3 * 0.7:
                                    return '8', float(confidence_8)
                    
                    # Case 2: If predicted as '6' or '9', check for possible confusion
                    elif predicted_char in ['6', '9']:
                        # Apply specialized 6 vs 9 distinction
                        corrected_digit = distinguish_6_vs_9(digit_img, predicted_char)
                        if corrected_digit != predicted_char:
                            # Find indices for digits '6' and '9' in the label mapping
                            idx_6 = next((i for i, v in self.label_mapping.items() if v == '6'), None)
                            idx_9 = next((i for i, v in self.label_mapping.items() if v == '9'), None)
                            
                            if idx_6 is not None and idx_9 is not None:
                                # Verify the correction with enhanced features
                                enhanced_features = enhance_digit_features(digit_img, '69')
                                enhanced_input = enhanced_features.reshape(1, 28, 28, 1)
                                
                                new_predictions = self.model.predict(enhanced_input, verbose=0)[0]
                                confidence_6 = new_predictions[idx_6]  # Confidence for digit 6
                                confidence_9 = new_predictions[idx_9]  # Confidence for digit 9
                                
                                # If confidence for the corrected digit is reasonable, use it
                                if corrected_digit == '6' and confidence_6 > confidence_9 * 0.6:
                                    return '6', float(confidence_6)
                                elif corrected_digit == '9' and confidence_9 > confidence_6 * 0.6:
                                    return '9', float(confidence_9)
                    
                    # Case 3: If predicted as '0' or '6', check for possible confusion
                    elif predicted_char in ['0', '6']:
                        # Apply specialized 0 vs 6 distinction
                        corrected_digit = distinguish_0_vs_6(digit_img, predicted_char)
                        if corrected_digit != predicted_char:
                            # Find indices for digits '0' and '6' in the label mapping
                            idx_0 = next((i for i, v in self.label_mapping.items() if v == '0'), None)
                            idx_6 = next((i for i, v in self.label_mapping.items() if v == '6'), None)
                            
                            if idx_0 is not None and idx_6 is not None:
                                # Verify the correction with enhanced features
                                enhanced_features = enhance_digit_features(digit_img, '06')
                                enhanced_input = enhanced_features.reshape(1, 28, 28, 1)
                                
                                new_predictions = self.model.predict(enhanced_input, verbose=0)[0]
                                confidence_0 = new_predictions[idx_0]  # Confidence for digit 0
                                confidence_6 = new_predictions[idx_6]  # Confidence for digit 6
                                
                                # If confidence for the corrected digit is reasonable, use it
                                if corrected_digit == '0' and confidence_0 > confidence_6 * 0.7:
                                    return '0', float(confidence_0)
                                elif corrected_digit == '6' and confidence_6 > confidence_0 * 0.7:
                                    return '6', float(confidence_6)
                    
                    # Case 4: If predicted as '1' or '7', check for possible confusion
                    elif predicted_char in ['1', '7']:
                        # Apply specialized 1 vs 7 distinction
                        corrected_digit = distinguish_1_vs_7(digit_img, predicted_char)
                        if corrected_digit != predicted_char:
                            # Find indices for digits '1' and '7' in the label mapping
                            idx_1 = next((i for i, v in self.label_mapping.items() if v == '1'), None)
                            idx_7 = next((i for i, v in self.label_mapping.items() if v == '7'), None)
                            
                            if idx_1 is not None and idx_7 is not None:
                                # Verify the correction with enhanced features
                                enhanced_features = enhance_digit_features(digit_img, '17')
                                enhanced_input = enhanced_features.reshape(1, 28, 28, 1)
                                
                                new_predictions = self.model.predict(enhanced_input, verbose=0)[0]
                                confidence_1 = new_predictions[idx_1]  # Confidence for digit 1
                                confidence_7 = new_predictions[idx_7]  # Confidence for digit 7
                                
                                # If confidence for the corrected digit is reasonable, use it
                                if corrected_digit == '1' and confidence_1 > confidence_7 * 0.7:
                                    return '1', float(confidence_1)
                                elif corrected_digit == '7' and confidence_7 > confidence_1 * 0.7:
                                    return '7', float(confidence_7)
            except Exception as e:
                print(f"Error in specialized digit classification: {e}")
            
            # Return the original prediction if no specialized classification applied
            return predicted_char, confidence
                
        except Exception as e:
            print(f"Error in prediction: {e}")
            return "", 0.0
    
    def get_top_predictions(self, image: np.ndarray, top_k: int = 3) -> list:
        """
        Get top K predictions with confidence scores
        
        Args:
            image: Input image
            top_k: Number of top predictions to return
            
        Returns:
            list: List of (character, confidence) tuples
        """
        if not self.is_loaded:
            return []
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            if processed_image is None:
                return []
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)[0]
            
            # Get top K predictions
            top_indices = np.argsort(predictions)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if idx in self.label_mapping:
                    char = self.label_mapping[idx]
                    confidence = float(predictions[idx])
                    results.append((char, confidence))
            
            return results
            
        except Exception as e:
            print(f"Error in top predictions: {e}")
            return []

# Global instance for the alphanumeric model
alphanumeric_model = AlphanumericModelLoader()

def load_alphanumeric_model() -> bool:
    """
    Load the alphanumeric model (convenience function)
    
    Returns:
        bool: True if successfully loaded
    """
    return alphanumeric_model.load_model()

def predict_alphanumeric_character(image: np.ndarray) -> Tuple[str, float]:
    """
    Predict alphanumeric character from image (convenience function)
    
    Args:
        image: Input image
        
    Returns:
        Tuple[str, float]: (predicted_character, confidence)
    """
    return alphanumeric_model.predict_character(image)

def get_alphanumeric_predictions(image: np.ndarray, top_k: int = 3) -> list:
    """
    Get top K alphanumeric predictions (convenience function)
    
    Args:
        image: Input image
        top_k: Number of predictions to return
        
    Returns:
        list: List of (character, confidence) tuples
    """
    return alphanumeric_model.get_top_predictions(image, top_k)

def is_alphanumeric_model_loaded() -> bool:
    """
    Check if alphanumeric model is loaded
    
    Returns:
        bool: True if model is loaded
    """
    return alphanumeric_model.is_loaded

# Test function
def test_alphanumeric_model():
    """Test the alphanumeric model with a sample image"""
    if not load_alphanumeric_model():
        print("Failed to load alphanumeric model for testing")
        return
    
    # Create a test image (28x28 with some pattern)
    test_image = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
    
    # Test prediction
    char, confidence = predict_alphanumeric_character(test_image)
    print(f"Test prediction: '{char}' (confidence: {confidence:.3f})")
    
    # Test top predictions
    top_preds = get_alphanumeric_predictions(test_image, top_k=5)
    print("Top 5 predictions:")
    for i, (c, conf) in enumerate(top_preds):
        print(f"  {i+1}. '{c}': {conf:.3f}")

if __name__ == "__main__":
    # Test the model loading and prediction
    print("=== Testing Alphanumeric Model Integration ===")
    test_alphanumeric_model()