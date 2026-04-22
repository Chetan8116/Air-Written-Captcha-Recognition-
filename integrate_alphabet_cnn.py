"""
Integration module for the new Alphabet CNN model from the alphabets directory
This module bridges the alphabet CNN model with the main VirtualPainter application
"""

import os
import numpy as np
import tensorflow as tf
import cv2
import json
from typing import Tuple, Dict, Optional, Union, List

class AlphabetCNNIntegrator:
    """Class to integrate the alphabet CNN model from the alphabets folder with the main application"""
    
    def __init__(self, 
                model_path: str = None,
                mapping_path: str = None):
        """
        Initialize the integration module
        
        Args:
            model_path: Path to the trained model file
            mapping_path: Path to the class mapping file
        """
        self.model = None
        self.class_mapping = {}
        self.model_loaded = False
        
        # Use default paths if none provided
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, "alphabets", "models")
        
        # Use the provided path or the default
        if model_path is None:
            # Default to the converted model
            model_path = os.path.join(base_dir, "models", "compatible", "alphabet_model_converted.h5")
        
        if mapping_path is None:
            # Default to the mapping in the alphabets/models directory
            mapping_path = os.path.join(models_dir, "alphabet_class_mapping.json")
        
        # Try to load the model
        if os.path.exists(model_path):
            print(f"Found alphabet CNN model at {model_path}")
            self.load_model(model_path, mapping_path)
        else:
            print(f"Alphabet CNN model not found at {model_path}.")
            print("Please run setup_alphabet_venv.bat to convert the model.")
    
    def load_model(self, model_path: str, mapping_path: str = None) -> bool:
        """
        Load the trained model and class mapping
        
        Args:
            model_path: Path to the trained model file
            mapping_path: Path to the class mapping file
            
        Returns:
            bool: True if loading was successful, False otherwise
        """
        try:
            print(f"Loading alphabet CNN model from {model_path}...")
            
            # Load the model (expecting a properly converted model)
            try:
                self.model = tf.keras.models.load_model(
                    model_path, 
                    compile=False  # Skip compilation to avoid optimizer issues
                )
                print("Model loaded successfully")
            except Exception as e:
                print(f"Model loading failed: {e}")
                print("Please run setup_alphabet_venv.bat to convert the model to a compatible format")
                return False

            # Load class mapping if available
            if mapping_path and os.path.exists(mapping_path):
                with open(mapping_path, 'r') as f:
                    self.class_mapping = json.load(f)
                print(f"Class mapping loaded with {len(self.class_mapping)} classes")
            else:
                # Create default mapping for 52 classes (A-Z, a-z)
                print("Creating default class mapping for 52 classes")
                for i in range(26):
                    self.class_mapping[str(i)] = chr(ord('A') + i)
                for i in range(26):
                    self.class_mapping[str(i + 26)] = chr(ord('a') + i)
            
            # Compile the model if needed
            if not self.model.compiled_loss:
                self.model.compile(
                    optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                print("Model compiled with default settings")
            
            self.model_loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading alphabet CNN model: {e}")
            self.model_loaded = False
            return False
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess an image for prediction
        
        Args:
            image: Input image (grayscale or BGR)
            
        Returns:
            np.ndarray: Processed image ready for model input
        """
        if image is None:
            return None
            
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Resize to 28x28
        resized = cv2.resize(gray, (28, 28))
        
        # Check if inversion is needed
        mean_pixel = np.mean(resized)
        if mean_pixel > 128:  # If background is light (our drawing is dark)
            resized = 255 - resized  # Invert
            
        # Normalize pixel values
        normalized = resized.astype('float32') / 255.0
        
        # Expand dimensions for model input
        expanded = np.expand_dims(normalized, axis=-1)  # Add channel dimension
        expanded = np.expand_dims(expanded, axis=0)  # Add batch dimension
        
        return expanded
    
    def predict_letter(self, image: np.ndarray, case_sensitive: bool = True) -> Tuple[str, float, Dict]:
        """
        Predict letter from the image
        
        Args:
            image: Input image (grayscale or BGR)
            case_sensitive: Whether to differentiate between uppercase and lowercase
            
        Returns:
            Tuple containing:
                - Predicted character
                - Confidence score
                - Dictionary with additional prediction info
        """
        if not self.model_loaded or self.model is None:
            return None, 0.0, {"error": "Model not loaded"}
            
        # Preprocess image
        processed_img = self.preprocess_image(image)
        
        if processed_img is None:
            return None, 0.0, {"error": "Image preprocessing failed"}
            
        # Get prediction
        try:
            # Make prediction with the model
            predictions = self.model.predict(processed_img, verbose=0)
            
            # Ensure predictions is a 1D array
            if len(predictions.shape) > 1:
                predictions = predictions[0]
                
            # Get top predicted class
            top_class = np.argmax(predictions)
            confidence = float(predictions[top_class])
            
            # Get letter from class mapping
            letter = self.class_mapping.get(str(top_class), '?')
                
            # Handle case sensitivity if requested
            if not case_sensitive and letter.isalpha():
                # Always return lowercase if case insensitive is requested
                letter = letter.lower()
            
            # Get top 5 predictions for detailed info
            top5_indices = np.argsort(predictions)[-5:][::-1]
            top5_predictions = {
                self.class_mapping.get(str(idx), f'Class_{idx}'): float(predictions[idx])
                for idx in top5_indices
            }
            
            # Return prediction info
            result = {
                "top5": top5_predictions,
                "class_index": int(top_class),
                "case_sensitive": case_sensitive
            }
            
            return letter, confidence, result
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return "?", 0.0, {"error": f"Prediction error: {e}"}
    
    def is_loaded(self) -> bool:
        """
        Check if the model is loaded
        
        Returns:
            bool: True if model is loaded, False otherwise
        """
        return self.model_loaded
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model
        
        Returns:
            Dict: Model information
        """
        if not self.model_loaded:
            return {"status": "Not loaded"}
            
        # Get model information
        info = {
            "status": "Loaded",
            "num_classes": len(self.class_mapping)
        }
        
        # Try to get input and output shapes safely
        try:
            info["input_shape"] = str(self.model.input_shape)
        except:
            info["input_shape"] = "unknown"
            
        try:
            info["output_shape"] = str(self.model.output_shape)
        except:
            info["output_shape"] = "unknown"
            
        # Try to get model summary as string
        try:
            import io
            summary_str = io.StringIO()
            self.model.summary(print_fn=lambda x: summary_str.write(x + '\n'))
            info["summary"] = summary_str.getvalue()
        except:
            info["summary"] = "Could not generate summary"
        
        return info

# Global instance for direct import
_instance = None

"""
Removed fallback model creation function as we now require the properly converted model
"""

def get_integrator() -> AlphabetCNNIntegrator:
    """
    Get the global integrator instance
    
    Returns:
        AlphabetCNNIntegrator: The global integrator instance
    """
    global _instance
    if _instance is None:
        _instance = AlphabetCNNIntegrator()
    return _instance

def load_alphabet_cnn_model(model_path: str = None, mapping_path: str = None) -> bool:
    """
    Load the alphabet CNN model
    
    Args:
        model_path: Path to the trained model file
        mapping_path: Path to the class mapping file
        
    Returns:
        bool: True if loading was successful, False otherwise
    """
    integrator = get_integrator()
    if model_path or not integrator.is_loaded():
        return integrator.load_model(model_path, mapping_path)
    return integrator.is_loaded()

def predict_with_alphabet_cnn(image: np.ndarray, case_sensitive: bool = True) -> Tuple[str, float, Dict]:
    """
    Predict letter from the image using the alphabet CNN model
    
    Args:
        image: Input image (grayscale or BGR)
        case_sensitive: Whether to differentiate between uppercase and lowercase
        
    Returns:
        Tuple containing:
            - Predicted character
            - Confidence score
            - Dictionary with additional prediction info
    """
    integrator = get_integrator()
    return integrator.predict_letter(image, case_sensitive)

def is_alphabet_cnn_loaded() -> bool:
    """
    Check if the alphabet CNN model is loaded
    
    Returns:
        bool: True if model is loaded, False otherwise
    """
    integrator = get_integrator()
    return integrator.is_loaded()