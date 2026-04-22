from enhanced_numeric_captcha import EnhancedCaptchaGenerator
from flask import session
import tensorflow as tf
import numpy as np
import os
import cv2
import random
import string
from PIL import Image
from io import BytesIO
import base64

# Import the comprehensive alphanumeric CAPTCHA classes
try:
    from comprehensive_alphanumeric_captcha import ComprehensiveAlphanumericGenerator
    COMPREHENSIVE_AVAILABLE = True
except ImportError:
    COMPREHENSIVE_AVAILABLE = False
    print("Comprehensive alphanumeric CAPTCHA module not available")

# Create global instances of the generators
enhanced_generator = None
realistic_model = None
comprehensive_generator = None

# Define different CAPTCHA modes
CAPTCHA_MODE_NUMERIC = 'numeric'
CAPTCHA_MODE_LOWERCASE = 'lowercase'
CAPTCHA_MODE_UPPERCASE = 'uppercase'
CAPTCHA_MODE_ALPHA = 'alpha'  # mixed case letters
CAPTCHA_MODE_ALPHANUMERIC = 'alphanumeric'  # mixed case letters and numbers

def get_enhanced_generator():
    """Get or create the enhanced captcha generator instance"""
    global enhanced_generator
    if enhanced_generator is None:
        try:
            enhanced_generator = EnhancedCaptchaGenerator()
        except Exception as e:
            print(f"Error initializing enhanced captcha generator: {e}")
            return None
    return enhanced_generator

def load_realistic_captcha_model():
    """Load the realistic captcha model trained from H5 file"""
    global realistic_model
    if realistic_model is None:
        model_path = os.path.join("realistic_numeric_captchas", "best_model.h5")
        if os.path.exists(model_path):
            try:
                realistic_model = tf.keras.models.load_model(model_path)
                print(f"Successfully loaded realistic captcha model from {model_path}")
                return realistic_model
            except Exception as e:
                print(f"Error loading realistic captcha model: {e}")
                return None
        else:
            print(f"Realistic captcha model not found at {model_path}")
            return None
    return realistic_model

def generate_enhanced_numeric_captcha(length=6, difficulty="medium"):
    """Generate an enhanced numeric captcha using the new generator"""
    generator = get_enhanced_generator()
    if generator is None:
        # Fallback to original captcha generator if enhanced one fails
        from captcha_utils import generate_numeric_captcha, generate_captcha_image
        text = generate_numeric_captcha(length)
        img_b64 = generate_captcha_image(text, "modern")
        return img_b64, text
    
    # Use the enhanced generator with safe call pattern
    try:
        img_b64, text = generator.generate_captcha_b64(length, difficulty)
    except Exception as e:
        print(f"Error generating enhanced captcha: {e}")
        # Fallback to original captcha generator
        from captcha_utils import generate_numeric_captcha, generate_captcha_image
        text = generate_numeric_captcha(length)
        img_b64 = generate_captcha_image(text, "modern")
    
    return img_b64, text

def get_comprehensive_generator():
    """Get or create the comprehensive alphanumeric captcha generator instance"""
    global comprehensive_generator
    if comprehensive_generator is None and COMPREHENSIVE_AVAILABLE:
        try:
            comprehensive_generator = ComprehensiveAlphanumericGenerator()
            print("Initialized comprehensive alphanumeric CAPTCHA generator")
        except Exception as e:
            print(f"Error initializing comprehensive alphanumeric CAPTCHA generator: {e}")
            return None
    return comprehensive_generator

def generate_realistic_numeric_captcha(length=6):
    """Generate a realistic numeric captcha using the trained model from H5"""
    # Try to use the enhanced generator with realistic settings
    generator = get_enhanced_generator()
    
    if generator is None:
        # Fallback to original captcha generator
        from captcha_utils import generate_numeric_captcha, generate_captcha_image
        text = generate_numeric_captcha(length)
        img_b64 = generate_captcha_image(text, "modern")
        return img_b64, text
    
    # Select a random difficulty level, with a bias toward harder captchas
    difficulties = ["easy", "medium", "hard"]
    weights = [0.2, 0.5, 0.3]  # 20% easy, 50% medium, 30% hard
    difficulty = random.choices(difficulties, weights=weights)[0]
    
    # Use the enhanced generator with the selected difficulty
    try:
        # Try with the additional parameters - for newer versions that support them
        img_b64, text = generator.generate_captcha_b64(length, difficulty, 
                                                   use_distortion=True,
                                                   use_perspective=True, 
                                                   use_noise=True)
    except TypeError:
        # Fallback for older versions that don't support these parameters
        img_b64, text = generator.generate_captcha_b64(length, difficulty)
        
    return img_b64, text

def generate_comprehensive_captcha(mode=CAPTCHA_MODE_ALPHANUMERIC, length=6, difficulty='medium'):
    """Generate a captcha using the comprehensive alphanumeric generator"""
    # Get the comprehensive generator
    generator = get_comprehensive_generator()
    
    if generator is None:
        # Fallback to original captcha generator if comprehensive one fails
        from captcha_utils import generate_numeric_captcha, generate_alpha_captcha, generate_alphanumeric_captcha, generate_captcha_image
        
        if mode == CAPTCHA_MODE_NUMERIC:
            text = generate_numeric_captcha(length)
        elif mode == CAPTCHA_MODE_LOWERCASE:
            text = generate_alpha_captcha(length, mode='lowercase')
        elif mode == CAPTCHA_MODE_UPPERCASE:
            text = generate_alpha_captcha(length, mode='uppercase')
        elif mode == CAPTCHA_MODE_ALPHA:
            text = generate_alpha_captcha(length, mode='mixed')
        else:  # alphanumeric
            # Ensure at least 2 digits in alphanumeric mode
            while True:
                text = generate_alphanumeric_captcha(length)
                digit_count = sum(1 for c in text if c.isdigit())
                if digit_count >= 2:
                    break
            
        img_b64 = generate_captcha_image(text, None)  # Use random style
        return img_b64, text
    
    # Use the comprehensive generator
    try:
        img_b64, text = generator.generate_captcha(mode=mode, length=length, difficulty=difficulty)
        
        # For alphanumeric mode, double-check that there are actually digits in the text
        if mode == CAPTCHA_MODE_ALPHANUMERIC:
            # If no digits or not enough digits, regenerate
            digit_count = sum(1 for c in text if c.isdigit())
            if digit_count < 2:
                # Try once more with explicit request for alphanumeric with digits
                img_b64, text = generator.generate_captcha(mode=mode, length=length, difficulty=difficulty)
                
        return img_b64, text
    except Exception as e:
        print(f"Error generating comprehensive captcha: {e}")
        # Fallback to original captcha generator
        from captcha_utils import generate_numeric_captcha, generate_alpha_captcha, generate_alphanumeric_captcha, generate_captcha_image
        
        if mode == CAPTCHA_MODE_NUMERIC:
            text = generate_numeric_captcha(length)
        elif mode == CAPTCHA_MODE_LOWERCASE:
            text = generate_alpha_captcha(length, mode='lowercase')
        elif mode == CAPTCHA_MODE_UPPERCASE:
            text = generate_alpha_captcha(length, mode='uppercase')
        elif mode == CAPTCHA_MODE_ALPHA:
            text = generate_alpha_captcha(length, mode='mixed')
        else:  # alphanumeric
            text = generate_alphanumeric_captcha(length)
            
        img_b64 = generate_captcha_image(text, None)
        return img_b64, text
    
    return img_b64, text

def get_captcha_for_current_mode(captcha_type=None):
    """Generate a captcha based on the current mode or specified type"""
    # Get the captcha type from session or use provided type
    if captcha_type is None:
        captcha_type = session.get('captcha_type', 'numeric')
    
    # Import here to avoid circular imports
    from VirtualPainter import get_current_recognition
    
    # Get current recognition settings
    current_recognition = get_current_recognition()
    current_mode = current_recognition["mode"]
    alphabet_mode = current_recognition.get("alphabet_mode", "auto")
    
    # Check if we should use comprehensive generator
    use_comprehensive = session.get('use_comprehensive_captcha', True)
    
    # Select the appropriate difficulty
    difficulties = ["easy", "medium", "hard"]
    weights = [0.3, 0.4, 0.3]  # 30% easy, 40% medium, 30% hard
    difficulty = random.choices(difficulties, weights=weights)[0]
    
    # Choose the appropriate captcha generation method
    if use_comprehensive and get_comprehensive_generator() is not None:
        # Use the comprehensive generator with the appropriate mode
        if current_mode in ['alpha', 'uppercase', 'lowercase', 'auto_alphabet'] or captcha_type == 'alpha':
            # Determine which case to use for alphabets (letters only)
            if current_mode == 'uppercase' or alphabet_mode == 'uppercase':
                captcha_mode = CAPTCHA_MODE_UPPERCASE
            elif current_mode == 'lowercase' or alphabet_mode == 'lowercase':
                captcha_mode = CAPTCHA_MODE_LOWERCASE
            else:
                captcha_mode = CAPTCHA_MODE_ALPHA  # Mixed case
                
            img_b64, text = generate_comprehensive_captcha(mode=captcha_mode, length=6, difficulty=difficulty)
            session["alpha_captcha"] = text
            session["captcha_type"] = "alpha"
            
        elif current_mode == 'alphanum' or captcha_type == 'alphanum':
                # For alphanumeric with the comprehensive generator
                # Always ensure we're using the alphanumeric mode which contains both letters and numbers
                # but respect the case preference for the letters
                if alphabet_mode == 'uppercase':
                    # Use explicit mode with case specification
                    mode_with_case = "alphanumeric-uppercase"
                    img_b64, text = generate_comprehensive_captcha(mode=mode_with_case, length=6, difficulty=difficulty)
                    
                    # Force all letters to uppercase while preserving digits
                    text = ''.join(c.upper() if c.isalpha() else c for c in text)
                    
                    # Double-check digit count - ensure we have at least 2 digits
                    digit_count = sum(1 for c in text if c.isdigit())
                    if digit_count < 2:
                        # Create a new text with guaranteed digits
                        digits = ''.join(random.choices(string.digits, k=2))  # At least 2 digits
                        letters = ''.join(random.choices(string.ascii_uppercase, k=4))
                        
                        combined = digits + letters
                        text_list = list(combined)
                        random.shuffle(text_list)
                        text = ''.join(text_list)
                        
                        # Regenerate the image with the new text
                        img_b64 = generate_captcha_image(text, None)
                    
                elif alphabet_mode == 'lowercase':
                    # Use explicit mode with case specification
                    mode_with_case = "alphanumeric-lowercase"
                    img_b64, text = generate_comprehensive_captcha(mode=mode_with_case, length=6, difficulty=difficulty)
                    
                    # Force all letters to lowercase while preserving digits
                    text = ''.join(c.lower() if c.isalpha() else c for c in text)
                    
                    # Double-check digit count - ensure we have at least 2 digits
                    digit_count = sum(1 for c in text if c.isdigit())
                    if digit_count < 2:
                        # Create a new text with guaranteed digits
                        digits = ''.join(random.choices(string.digits, k=2))  # At least 2 digits
                        letters = ''.join(random.choices(string.ascii_lowercase, k=4))
                        
                        combined = digits + letters
                        text_list = list(combined)
                        random.shuffle(text_list)
                        text = ''.join(text_list)
                        
                        # Regenerate the image with the new text
                        img_b64 = generate_captcha_image(text, None)
                    
                else:
                    # Mixed case alphanumeric
                    img_b64, text = generate_comprehensive_captcha(mode=CAPTCHA_MODE_ALPHANUMERIC, length=6, difficulty=difficulty)
                    
                    # Double-check digit count - ensure we have at least 2 digits
                    digit_count = sum(1 for c in text if c.isdigit())
                    if digit_count < 2:
                        # Create a new text with guaranteed digits
                        digits = ''.join(random.choices(string.digits, k=2))  # At least 2 digits
                        letters = ''.join(random.choices(string.ascii_letters, k=4))
                        
                        combined = digits + letters
                        text_list = list(combined)
                        random.shuffle(text_list)
                        text = ''.join(text_list)
                        
                        # Regenerate the image with the new text
                        img_b64 = generate_captcha_image(text, None)
                
                session["alphanum_captcha"] = text
                session["captcha_type"] = "alphanum"
        else:
            # For numeric, check if we should use realistic captcha
            use_realistic = session.get('use_realistic_captcha', True)  # Default to realistic
            
            if use_realistic and captcha_type != 'enhanced_numeric':
                # Use the new realistic captcha model
                img_b64, text = generate_realistic_numeric_captcha(6)
                session["numeric_captcha"] = text
                session["captcha_type"] = "realistic_numeric"
            else:
                # Use comprehensive generator in numeric mode
                img_b64, text = generate_comprehensive_captcha(mode=CAPTCHA_MODE_NUMERIC, length=6, difficulty=difficulty)
                session["numeric_captcha"] = text
                session["captcha_type"] = "comprehensive_numeric"
    else:
        # Use the legacy generators
        from captcha_utils import generate_alpha_captcha, generate_alphanumeric_captcha, generate_captcha_image
        
        if current_mode in ['alpha', 'uppercase', 'lowercase', 'auto_alphabet'] or captcha_type == 'alpha':
            # Determine which case to use for alphabets
            if current_mode == 'uppercase' or alphabet_mode == 'uppercase':
                captcha_mode = 'uppercase'
            elif current_mode == 'lowercase' or alphabet_mode == 'lowercase':
                captcha_mode = 'lowercase'
            else:
                captcha_mode = 'auto'  # Mixed case
                
            text = generate_alpha_captcha(6, mode=captcha_mode)
            img_b64 = generate_captcha_image(text, None)  # Use random style
            session["alpha_captcha"] = text
            session["captcha_type"] = "alpha"
            
        elif current_mode == 'alphanum' or captcha_type == 'alphanum':
            # For alphanumeric, check the alphabet mode
            if alphabet_mode == 'uppercase':
                text = generate_alphanumeric_captcha(6, mode='uppercase')
            elif alphabet_mode == 'lowercase':
                text = generate_alphanumeric_captcha(6, mode='lowercase')
            else:
                text = generate_alphanumeric_captcha(6)
            
            # Verify that the text contains at least two digits
            digit_count = sum(1 for c in text if c.isdigit())
            if digit_count < 2:
                # If not enough digits, manually insert 2 digits at random positions
                num_digits_to_add = 2 - digit_count
                
                # Find letter positions where we can replace with digits
                letter_positions = [i for i, c in enumerate(text) if c.isalpha()]
                
                if len(letter_positions) >= num_digits_to_add:
                    positions_to_replace = random.sample(letter_positions, num_digits_to_add)
                    text_list = list(text)
                    
                    for pos in positions_to_replace:
                        text_list[pos] = random.choice(string.digits)
                    
                    text = ''.join(text_list)
                else:
                    # If not enough letters to replace, just generate a new one with enforced digit count
                    digits = ''.join(random.choices(string.digits, k=2))  # At least 2 digits
                    
                    if alphabet_mode == 'uppercase':
                        letters = ''.join(random.choices(string.ascii_uppercase, k=4))
                    elif alphabet_mode == 'lowercase':
                        letters = ''.join(random.choices(string.ascii_lowercase, k=4))
                    else:
                        letters = ''.join(random.choices(string.ascii_letters, k=4))
                    
                    combined = digits + letters
                    text_list = list(combined)
                    random.shuffle(text_list)
                    text = ''.join(text_list)
                
            img_b64 = generate_captcha_image(text, None)  # Use random style
            session["alphanum_captcha"] = text
            session["captcha_type"] = "alphanum"
        else:
            # For numeric, check if we should use realistic captcha
            use_realistic = session.get('use_realistic_captcha', True)  # Default to realistic
            
            if use_realistic and captcha_type != 'enhanced_numeric':
                # Use the new realistic captcha model
                img_b64, text = generate_realistic_numeric_captcha(6)
                session["numeric_captcha"] = text
                session["captcha_type"] = "realistic_numeric"
            else:
                # Fallback to enhanced generator
                img_b64, text = generate_enhanced_numeric_captcha(6, "medium")
                session["numeric_captcha"] = text
                session["captcha_type"] = "numeric"
    
    return img_b64, text