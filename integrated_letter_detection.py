"""
Integrated letter detection system for VirtualPainter
Combines specialized letter detectors, neural network models, and structural analysis
"""

import os
import sys
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import traceback

# Import specialized letter detector
try:
    from specialized_letter_detector import detect_specific_letter, detect_all_letters
    SPECIALIZED_DETECTORS_AVAILABLE = True
except ImportError as e:
    print(f"Specialized letter detectors not available: {e}")
    SPECIALIZED_DETECTORS_AVAILABLE = False

# Dictionary to store loaded models
LETTER_DETECTOR_MODELS = {}

def load_letter_models():
    """Load all available letter detector models"""
    global LETTER_DETECTOR_MODELS, SPECIALIZED_DETECTORS_AVAILABLE
    
    if not SPECIALIZED_DETECTORS_AVAILABLE:
        print("Specialized letter detectors not available")
        return False
    
    # Check models directory
    models_dir = "models/specialized"
    if not os.path.exists(models_dir):
        print(f"Models directory not found: {models_dir}")
        os.makedirs(models_dir, exist_ok=True)
        return False
    
    # Track which models were loaded
    loaded_models = []
    failed_models = []
    
    # Try to load models for all letters A-Z
    for letter_code in range(ord('A'), ord('Z') + 1):
        letter = chr(letter_code)
        model_path = os.path.join(models_dir, f"{letter.lower()}_detector.h5")
        
        if os.path.exists(model_path):
            try:
                # Load the model
                model = keras.models.load_model(model_path)
                LETTER_DETECTOR_MODELS[letter] = model
                loaded_models.append(letter)
                print(f"Loaded detector model for letter {letter}")
            except Exception as e:
                print(f"Error loading model for letter {letter}: {e}")
                failed_models.append(letter)
        else:
            print(f"Model not found for letter {letter}")
            failed_models.append(letter)
    
    # Report results
    if loaded_models:
        print(f"Successfully loaded models for letters: {', '.join(loaded_models)}")
    
    if failed_models:
        print(f"Failed to load models for letters: {', '.join(failed_models)}")
    
    return len(loaded_models) > 0

def is_model_loaded(letter):
    """Check if a specific letter model is loaded"""
    global LETTER_DETECTOR_MODELS
    return letter in LETTER_DETECTOR_MODELS and LETTER_DETECTOR_MODELS[letter] is not None

def detect_letter(image, letter, threshold=0.5):
    """Detect if an image contains a specific letter"""
    global LETTER_DETECTOR_MODELS, SPECIALIZED_DETECTORS_AVAILABLE
    
    # Try using the specialized detector first
    if SPECIALIZED_DETECTORS_AVAILABLE:
        try:
            is_letter, confidence = detect_specific_letter(image, letter, threshold=threshold)
            if is_letter:
                return True, confidence
        except Exception as e:
            print(f"Error in specialized detector for {letter}: {e}")
    
    # Fall back to direct model prediction if available
    if letter in LETTER_DETECTOR_MODELS and LETTER_DETECTOR_MODELS[letter] is not None:
        try:
            # Preprocess the image
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Resize to 28x28
            resized = cv2.resize(gray, (28, 28))
            
            # Normalize and reshape for prediction
            normalized = resized.astype('float32') / 255.0
            input_img = normalized.reshape(1, 28, 28, 1)
            
            # Predict
            prediction = LETTER_DETECTOR_MODELS[letter].predict(input_img, verbose=0)[0]
            is_letter = prediction[1] > threshold
            confidence = float(prediction[1])
            
            return is_letter, confidence
        except Exception as e:
            print(f"Error in direct model prediction for {letter}: {e}")
            traceback.print_exc()
    
    return False, 0.0

def analyze_all_letters(image, min_confidence=0.5, uppercase_only=True):
    """Analyze image for all letters and return the best match"""
    global SPECIALIZED_DETECTORS_AVAILABLE
    
    # First try the integrated detect_all_letters function if available
    if SPECIALIZED_DETECTORS_AVAILABLE:
        try:
            best_letter, best_confidence, letter_scores = detect_all_letters(image, min_confidence=min_confidence)
            if best_letter and best_confidence >= min_confidence:
                print(f"Specialized detector found letter {best_letter} with confidence {best_confidence:.4f}")
                return best_letter, best_confidence, letter_scores
        except Exception as e:
            print(f"Error in detect_all_letters: {e}")
    
    # Fall back to checking each letter model individually
    best_letter = None
    best_confidence = 0.0
    letter_scores = {}
    
    letter_range = range(ord('A'), ord('Z') + 1) if uppercase_only else range(ord('A'), ord('z') + 1)
    
    for letter_code in letter_range:
        letter = chr(letter_code)
        # Skip lowercase if uppercase only
        if uppercase_only and letter.islower():
            continue
            
        is_letter, confidence = detect_letter(image, letter, threshold=min_confidence)
        letter_scores[letter] = confidence
        
        # Update best match if necessary
        if is_letter and confidence > best_confidence:
            best_letter = letter
            best_confidence = confidence
    
    return best_letter, best_confidence, letter_scores

def get_confused_letter_pairs():
    """Return pairs of letters that are commonly confused"""
    confused_pairs = [
        ('B', 'D'), ('B', 'P'), ('B', 'R'), ('B', '8'),
        ('C', 'G'), ('C', 'O'), 
        ('D', 'O'), ('D', 'P'),
        ('E', 'F'), ('E', '3'),
        ('G', 'Q'), ('G', '6'),
        ('I', 'J'), ('I', '1'), ('I', 'L'),
        ('K', 'X'), ('K', 'R'),
        ('M', 'N'), ('M', 'W'),
        ('N', 'Z'),
        ('O', 'Q'), ('O', '0'), ('O', 'D'),
        ('P', 'R'),
        ('S', '5'), ('S', '8'),
        ('T', 'I'), ('T', '7'),
        ('U', 'V'), ('U', 'Y'),
        ('V', 'W'),
        ('X', 'K'), ('X', 'Y'),
        ('Z', '2'), ('Z', '7')
    ]
    return confused_pairs

def resolve_confused_letters(letter, confidence, image):
    """For commonly confused letter pairs, run additional analysis to determine the correct letter"""
    confused_pairs = get_confused_letter_pairs()
    
    # Check if our letter is in any confused pairs
    potentially_confused = []
    for pair in confused_pairs:
        if letter in pair:
            # Get the other letter in the pair
            other_letter = pair[1] if letter == pair[0] else pair[0]
            potentially_confused.append(other_letter)
    
    if not potentially_confused:
        return letter, confidence
    
    print(f"Letter {letter} might be confused with {potentially_confused}")
    
    # Check all potentially confused letters
    best_letter = letter
    best_confidence = confidence
    
    for confused_letter in potentially_confused:
        # Only check letters (not digits)
        if confused_letter.isalpha():
            is_letter, conf = detect_letter(image, confused_letter, threshold=0.3)
            print(f"Checking if might be {confused_letter}: {is_letter} ({conf:.4f})")
            
            if is_letter and conf > best_confidence + 0.1:  # Must be significantly more confident
                best_letter = confused_letter
                best_confidence = conf
    
    return best_letter, best_confidence

def detect_letter_with_color_analysis(image, letter):
    """Detect letter with additional color analysis"""
    # Check for specific color characteristics (like red B)
    has_color = False
    color_name = "unknown"
    
    try:
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Split into color channels
            b, g, r = cv2.split(image)
            
            # Calculate average intensities
            avg_r = np.mean(r)
            avg_g = np.mean(g)
            avg_b = np.mean(b)
            
            # Check for dominant colors
            if avg_r > avg_g * 1.5 and avg_r > avg_b * 1.5:
                has_color = True
                color_name = "red"
            elif avg_g > avg_r * 1.5 and avg_g > avg_b * 1.5:
                has_color = True
                color_name = "green"
            elif avg_b > avg_r * 1.5 and avg_b > avg_g * 1.5:
                has_color = True
                color_name = "blue"
    except Exception as e:
        print(f"Error in color analysis: {e}")
    
    # Run the letter detection
    is_letter, confidence = detect_letter(image, letter)
    
    # For specific letter-color combinations, adjust confidence
    if has_color and is_letter:
        # Red B gets a confidence boost
        if letter == 'B' and color_name == "red":
            confidence = min(1.0, confidence * 1.2)
            print(f"Boosting confidence for red B to {confidence:.4f}")
    
    return is_letter, confidence, has_color, color_name

# Main integration function
def analyze_and_detect_letter(image, uppercase_only=True):
    """Complete analysis to detect the most likely letter in the image"""
    
    # First try the comprehensive detection
    best_letter, confidence, scores = analyze_all_letters(image, min_confidence=0.3, uppercase_only=uppercase_only)
    
    if best_letter and confidence >= 0.5:
        # High confidence detection
        print(f"High confidence detection: {best_letter} ({confidence:.4f})")
        return best_letter, confidence
    
    elif best_letter and confidence >= 0.3:
        # Medium confidence - check for commonly confused letters
        resolved_letter, resolved_confidence = resolve_confused_letters(best_letter, confidence, image)
        
        if resolved_letter != best_letter:
            print(f"Resolved confusion: {best_letter} -> {resolved_letter} ({resolved_confidence:.4f})")
            return resolved_letter, resolved_confidence
        
        # Try color analysis for extra confidence
        is_letter, color_confidence, has_color, color_name = detect_letter_with_color_analysis(image, best_letter)
        
        if has_color and color_confidence > confidence:
            print(f"Color analysis improved detection: {best_letter} ({color_confidence:.4f}, {color_name})")
            return best_letter, color_confidence
        
        return best_letter, confidence
    
    # Low confidence or no detection
    print("No confident letter detection")
    return None, 0.0

# Function to test the system
def test_letter_detection(image_path):
    """Test the letter detection system on a specific image"""
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return
    
    # Run the detection
    letter, confidence = analyze_and_detect_letter(image, uppercase_only=True)
    
    print(f"Final detection: {letter} with confidence {confidence:.4f}")
    
    # Display the result
    result_image = image.copy()
    cv2.putText(result_image, f"{letter}: {confidence:.2f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    # Save the result
    output_path = os.path.splitext(image_path)[0] + "_detection.jpg"
    cv2.imwrite(output_path, result_image)
    print(f"Detection result saved to {output_path}")

if __name__ == "__main__":
    # Load all models
    load_letter_models()
    
    # Test on sample image if provided
    if len(sys.argv) > 1:
        test_letter_detection(sys.argv[1])
    else:
        print("Usage: python integrated_letter_detection.py <image_path>")