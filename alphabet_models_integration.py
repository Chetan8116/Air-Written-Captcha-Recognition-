"""
Integration script for specialized alphabet models (uppercase and lowercase)
This module integrates models for uppercase (A-Z) and lowercase (a-z) letters
from the models folder, trained with EMNIST or custom datasets.
"""

import cv2
import numpy as np
import tensorflow as tf
import json
import os
from typing import Optional, Tuple, Dict, Any, List, Union

class AlphabetModelsLoader:
    """Class to load and manage both uppercase and lowercase alphabet models"""
    
    def __init__(self):
        self.uppercase_model = None
        self.lowercase_model = None
        self.uppercase_labels = {}
        self.lowercase_labels = {}
        self.uppercase_mapping = {}
        self.lowercase_mapping = {}
        self.is_uppercase_loaded = False
        self.is_lowercase_loaded = False
        
    def load_models(self, 
                  uppercase_model_path: str = "models/compatible/uppercase_model.h5",
                  lowercase_model_path: str = "models/compatible/lowercase_model.h5",  # Updated with new trained model
                  lowercase_labels_path: str = "models/compatible/lowercase_labels.json") -> bool:  # Updated with new trained model labels
        """
        Load both uppercase and lowercase alphabet models
        
        Args:
            uppercase_model_path: Path to the uppercase letters model
            lowercase_model_path: Path to the lowercase letters model
            lowercase_labels_path: Path to the lowercase label mappings file
            
        Returns:
            bool: True if at least one model loaded successfully, False otherwise
        """
        uppercase_loaded = self._load_uppercase_model(uppercase_model_path)
        lowercase_loaded = self._load_lowercase_model(lowercase_model_path, lowercase_labels_path)
        
        return uppercase_loaded or lowercase_loaded
    
    def _load_uppercase_model(self, model_path: str) -> bool:
        """Load the uppercase A-Z model"""
        try:
            if not os.path.exists(model_path):
                print(f"Warning: Uppercase model file '{model_path}' not found.")
                return False
            
            # Direct loading of our compatible model
            print(f"Loading uppercase alphabet model from {model_path}...")
            self.uppercase_model = tf.keras.models.load_model(model_path)
            print("Uppercase model loaded successfully!")
            
            # Create A-Z mapping (index to character)
            self.uppercase_mapping = {i: chr(ord('A') + i) for i in range(26)}
            self.is_uppercase_loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading uppercase model: {e}")
            return False
    
    def _load_lowercase_model(self, model_path: str, labels_path: str) -> bool:
        """Load the lowercase a-z model"""
        try:
            if not os.path.exists(model_path):
                print(f"Warning: Lowercase model file '{model_path}' not found.")
                return False
            
            # Use a straightforward mapping for lowercase letters
            # Create a mapping of the form {"0": "a", "1": "b", ...}
            # Force lowercase letters by using lowercase ASCII values
            self.lowercase_mapping = {str(i): chr(ord('a') + i) for i in range(26)}
            
            # Try loading the label file if it exists, but use our default if there are issues
            if os.path.exists(labels_path):
                try:
                    with open(labels_path, 'r') as f:
                        loaded_labels = json.load(f)
                    print(f"Loaded lowercase labels: {loaded_labels}")
                    # Only use loaded labels if they look valid
                    if isinstance(loaded_labels, dict) and len(loaded_labels) > 0:
                        # Convert all labels to lowercase to ensure lowercase output
                        self.lowercase_mapping = {k: v.lower() if isinstance(v, str) else v for k, v in loaded_labels.items()}
                except Exception as e:
                    print(f"Error loading lowercase labels, using default: {e}")
            
            # Direct loading of our compatible model
            print(f"Loading lowercase alphabet model from {model_path}...")
            self.lowercase_model = tf.keras.models.load_model(model_path)
            print("Lowercase model loaded successfully!")
            
            self.is_lowercase_loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading lowercase model: {e}")
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
        
        # Invert if needed (EMNIST models expect white characters on black background)
        mean_pixel = np.mean(resized)
        if mean_pixel > 128:  # If background is light (our drawing is dark)
            resized = 255 - resized  # Invert
        
        # Normalize pixel values
        normalized = resized.astype('float32') / 255.0
        
        # Expand dimensions for model input
        expanded = np.expand_dims(normalized, axis=-1)  # Add channel dimension
        expanded = np.expand_dims(expanded, axis=0)  # Add batch dimension
        
        return expanded
    
    def predict_letter(self, image: np.ndarray, mode: str = 'auto') -> Tuple[str, float, Dict]:
        """
        Predict letter from the image using the appropriate model
        
        Args:
            image: Preprocessed image
            mode: One of 'auto', 'uppercase', or 'lowercase'
            
        Returns:
            Tuple containing:
                - Predicted character
                - Confidence score
                - Dictionary with additional prediction info
        """
        if image is None:
            return None, 0.0, {}
        
        processed_img = self.preprocess_image(image)
        
        results = {}
        
        # Check if both models should be used
        if mode == 'auto':
            uppercase_pred = None
            lowercase_pred = None
            
            if self.is_uppercase_loaded:
                uppercase_pred = self.uppercase_model.predict(processed_img)[0]
                uppercase_class = np.argmax(uppercase_pred)
                uppercase_conf = float(uppercase_pred[uppercase_class])
                # Ensure uppercase character is actually uppercase
                uppercase_char = str(self.uppercase_mapping.get(uppercase_class, '?')).upper()
                results['uppercase'] = {
                    'char': uppercase_char,
                    'confidence': uppercase_conf,
                    'class_idx': int(uppercase_class)
                }
                
            if self.is_lowercase_loaded:
                lowercase_pred = self.lowercase_model.predict(processed_img)[0]
                lowercase_class = np.argmax(lowercase_pred)
                lowercase_conf = float(lowercase_pred[lowercase_class])
                lowercase_key = str(lowercase_class)
                # Ensure lowercase character is actually lowercase
                lowercase_char = str(self.lowercase_mapping.get(lowercase_key, '?')).lower()
                results['lowercase'] = {
                    'char': lowercase_char,
                    'confidence': lowercase_conf,
                    'class_idx': int(lowercase_class)
                }
            
            # Choose the prediction with higher confidence
            if uppercase_pred is not None and lowercase_pred is not None:
                if results['uppercase']['confidence'] > results['lowercase']['confidence'] * 1.1:  # Slight bias for uppercase
                    chosen_char = results['uppercase']['char']
                    confidence = results['uppercase']['confidence']
                    print(f"Auto mode: Selected uppercase {chosen_char} ({confidence:.4f}) over lowercase {results['lowercase']['char']} ({results['lowercase']['confidence']:.4f})")
                else:
                    chosen_char = results['lowercase']['char']
                    confidence = results['lowercase']['confidence']
                    print(f"Auto mode: Selected lowercase {chosen_char} ({confidence:.4f}) over uppercase {results['uppercase']['char']} ({results['uppercase']['confidence']:.4f})")
                return chosen_char, confidence, results
            elif uppercase_pred is not None:
                return results['uppercase']['char'], results['uppercase']['confidence'], results
            elif lowercase_pred is not None:
                return results['lowercase']['char'], results['lowercase']['confidence'], results
            else:
                return '?', 0.0, {}
                
        # Use only uppercase model
        elif mode == 'uppercase' and self.is_uppercase_loaded:
            pred = self.uppercase_model.predict(processed_img)[0]
            class_idx = np.argmax(pred)
            confidence = float(pred[class_idx])
            # Ensure uppercase character is actually uppercase
            char = str(self.uppercase_mapping.get(class_idx, '?')).upper()
            results = {
                'uppercase': {
                    'char': char,
                    'confidence': confidence,
                    'class_idx': int(class_idx)
                }
            }
            print(f"Uppercase mode prediction: {char} with confidence {confidence:.4f}")
            return char, confidence, results
            
        # Use only lowercase model
        elif mode == 'lowercase' and self.is_lowercase_loaded:
            pred = self.lowercase_model.predict(processed_img)[0]
            class_idx = np.argmax(pred)
            confidence = float(pred[class_idx])
            key = str(class_idx)
            # Ensure character is always lowercase
            char = str(self.lowercase_mapping.get(key, '?')).lower()
            results = {
                'lowercase': {
                    'char': char,
                    'confidence': confidence,
                    'class_idx': int(class_idx)
                }
            }
            print(f"Lowercase mode prediction: {char} with confidence {confidence:.4f}")
            return char, confidence, results
            
        # Fallback
        return '?', 0.0, {}
            
# Singleton instance
_alphabet_models = None

def load_alphabet_models() -> bool:
    """
    Load the specialized alphabet models
    
    Returns:
        bool: True if at least one model was loaded successfully
    """
    global _alphabet_models
    
    if _alphabet_models is None:
        _alphabet_models = AlphabetModelsLoader()
    
    return _alphabet_models.load_models()

def is_alphabet_models_loaded() -> bool:
    """
    Check if the specialized alphabet models are loaded
    
    Returns:
        bool: True if at least one model is loaded
    """
    global _alphabet_models
    
    if _alphabet_models is None:
        # Try loading the models
        load_alphabet_models()
        # If still None, return False
        if _alphabet_models is None:
            return False
        
    # Debug output to help diagnose issues
    print(f"Alphabet models status: uppercase={_alphabet_models.is_uppercase_loaded}, lowercase={_alphabet_models.is_lowercase_loaded}")
    
    return _alphabet_models.is_uppercase_loaded or _alphabet_models.is_lowercase_loaded

def predict_letter(image: np.ndarray, mode: str = 'auto') -> Tuple[str, float, Dict]:
    """
    Predict letter from the image
    
    Args:
        image: Input image
        mode: One of 'auto', 'uppercase', or 'lowercase'
        
    Returns:
        Tuple containing:
            - Predicted character
            - Confidence score
            - Dictionary with additional prediction info
    """
    global _alphabet_models
    
    print(f"predict_letter called with mode={mode}")
    
    if _alphabet_models is None:
        print("Alphabet models not loaded, trying to load now")
        load_alphabet_models()
    
    # Check if models are available
    if _alphabet_models is None:
        print("Failed to load alphabet models")
        return '?', 0.0, {}
        
    if not is_alphabet_models_loaded():
        print("No alphabet models loaded successfully")
        return '?', 0.0, {}
        
    print(f"Alphabet models status: uppercase={_alphabet_models.is_uppercase_loaded}, lowercase={_alphabet_models.is_lowercase_loaded}")
        
    return _alphabet_models.predict_letter(image, mode)

def get_available_alphabet_modes() -> List[str]:
    """
    Get available alphabet prediction modes
    
    Returns:
        List of available modes ('uppercase', 'lowercase', 'auto')
    """
    global _alphabet_models
    
    if _alphabet_models is None:
        load_alphabet_models()
        
    modes = ['auto']
    
    if _alphabet_models.is_uppercase_loaded:
        modes.append('uppercase')
        
    if _alphabet_models.is_lowercase_loaded:
        modes.append('lowercase')
        
    return modes