import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
import base64
from io import BytesIO
import string
import random

class ComprehensiveAlphanumericCAPTCHA:
    def __init__(self, model_path=None):
        """Initialize the comprehensive alphanumeric CAPTCHA recognizer"""
        if model_path is None:
            model_path = os.path.join("comprehensive_alphanumeric_dataset", "alphanumeric_captcha_model.h5")
            
        self.model_path = model_path
        self.model = None
        self.char_mappings = None
        self.config = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and configuration"""
        try:
            # Load model
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                print(f"Loaded comprehensive alphanumeric CAPTCHA model from {self.model_path}")
            else:
                print(f"Model file not found at {self.model_path}")
                return False
            
            # Load character mappings
            dataset_dir = os.path.dirname(self.model_path)
            mappings_path = os.path.join(dataset_dir, "char_mappings.json")
            
            if os.path.exists(mappings_path):
                with open(mappings_path, 'r') as f:
                    mappings = json.load(f)
                    # Convert index keys back to integers
                    self.char_mappings = {
                        'char_to_idx': mappings['char_to_idx'],
                        'idx_to_char': {int(k): v for k, v in mappings['idx_to_char'].items()}
                    }
            else:
                print(f"Character mappings file not found at {mappings_path}")
                # Create default mappings
                chars = string.digits + string.ascii_lowercase + string.ascii_uppercase
                self.char_mappings = {
                    'char_to_idx': {char: idx for idx, char in enumerate(chars)},
                    'idx_to_char': {idx: char for idx, char in enumerate(chars)}
                }
            
            # Load config
            config_path = os.path.join(dataset_dir, "model_config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            else:
                print(f"Config file not found at {config_path}")
                # Create default config
                self.config = {
                    'max_length': 6,
                    'num_classes': len(self.char_mappings['char_to_idx']),
                    'image_width': 200,
                    'image_height': 80
                }
                
            return True
        
        except Exception as e:
            print(f"Error loading comprehensive alphanumeric CAPTCHA model: {e}")
            return False
    
    def preprocess_image(self, image):
        """Preprocess image for prediction"""
        if isinstance(image, str) and image.startswith('data:image'):
            # Base64 image
            image = image.split(',')[1]
            image = BytesIO(base64.b64decode(image))
            image = Image.open(image)
        elif isinstance(image, str):
            # File path
            image = Image.open(image)
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Resize to expected dimensions
        image = image.resize((self.config['image_width'], self.config['image_height']))
        
        # Convert to numpy array and normalize
        img_array = np.array(image) / 255.0
        
        # Add batch and channel dimensions
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.expand_dims(img_array, axis=-1)
        
        return img_array
    
    def recognize(self, image):
        """Recognize characters in the CAPTCHA image"""
        if self.model is None:
            print("Model not loaded")
            return None
        
        # Preprocess image
        img_array = self.preprocess_image(image)
        
        # Predict
        predictions = self.model.predict(img_array)
        
        # Decode predictions
        result = ''
        for pred in predictions:
            char_idx = np.argmax(pred[0])
            char = self.char_mappings['idx_to_char'][char_idx]
            result += char
            
            # Stop at the first blank prediction
            if char == ' ':
                break
        
        return result.strip()

class ComprehensiveAlphanumericGenerator:
    def __init__(self, dataset_path="comprehensive_alphanumeric_dataset"):
        """Initialize the alphanumeric CAPTCHA generator"""
        from train_comprehensive_alphanumeric_captcha import ComprehensiveCaptchaGenerator
        self.generator = ComprehensiveCaptchaGenerator()
        
    def generate_captcha(self, mode='mixed', length=6, difficulty='medium'):
        """Generate a CAPTCHA with the specified mode and difficulty"""
        # Available modes: numeric, lowercase, uppercase, alpha, alphanumeric/mixed
        
        # Generate text based on the mode
        if mode == 'numeric':
            text = ''.join(random.choice(string.digits) for _ in range(length))
        elif mode == 'lowercase':
            text = ''.join(random.choice(string.ascii_lowercase) for _ in range(length))
        elif mode == 'uppercase':
            text = ''.join(random.choice(string.ascii_uppercase) for _ in range(length))
        elif mode == 'alpha':
            text = ''.join(random.choice(string.ascii_letters) for _ in range(length))
        elif mode == 'alphanumeric' or mode == 'mixed' or (isinstance(mode, str) and mode.startswith('alphanumeric')):
            # Make sure there's a mix of letters and numbers for alphanumeric mode
            # Always include at least 2 digits (or more) to ensure it's visibly alphanumeric
            num_digits = random.randint(2, max(2, length // 2))  # At least 2, at most half the length
            num_letters = length - num_digits
            
            # Determine the letter case based on mode specifics
            if isinstance(mode, str) and "-" in mode:
                case_mode = mode.split("-")[1]
                if case_mode == 'uppercase':
                    letters = string.ascii_uppercase
                elif case_mode == 'lowercase':
                    letters = string.ascii_lowercase
                else:
                    letters = string.ascii_letters
            else:
                # Default to mixed case
                letters = string.ascii_letters
                
            # Generate digits - always include at least 2
            digits_part = ''.join(random.choice(string.digits) for _ in range(num_digits))
            # Generate letters with appropriate case
            letters_part = ''.join(random.choice(letters) for _ in range(num_letters))
            
            # Combine and shuffle
            combined = digits_part + letters_part
            combined_list = list(combined)
            random.shuffle(combined_list)
            text = ''.join(combined_list)
        else:
            # Default to mixed alphanumeric
            # Ensure there are at least 2 digits
            num_digits = random.randint(2, max(2, length // 2))
            num_letters = length - num_digits
            
            digits_part = ''.join(random.choice(string.digits) for _ in range(num_digits))
            letters_part = ''.join(random.choice(string.ascii_letters) for _ in range(num_letters))
            
            combined = digits_part + letters_part
            combined_list = list(combined)
            random.shuffle(combined_list)
            text = ''.join(combined_list)
            
        # Generate image
        image, _ = self.generator.generate_captcha(text=text, mode=mode, difficulty=difficulty)
        
        # Convert to base64
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return img_str, text
    
    def generate_captcha_b64(self, mode='mixed', length=6, difficulty='medium'):
        """Generate a CAPTCHA and return as base64 string"""
        return self.generate_captcha(mode=mode, length=length, difficulty=difficulty)

def integrate_with_app():
    """Instructions to integrate with the main app"""
    print("To integrate this comprehensive alphanumeric CAPTCHA with your app:")
    print("1. Import the classes from this module in enhanced_captcha_integration.py")
    print("2. Create an instance of ComprehensiveAlphanumericGenerator for generating CAPTCHAs")
    print("3. Use the generator to create CAPTCHAs with different modes (numeric, lowercase, uppercase, etc.)")
    print("4. Update the get_captcha_for_current_mode function to use the appropriate mode based on settings")

if __name__ == "__main__":
    # Test the generator
    print("Testing comprehensive alphanumeric CAPTCHA generator...")
    
    generator = ComprehensiveAlphanumericGenerator()
    
    # Test with different modes
    modes = ['numeric', 'lowercase', 'uppercase', 'alpha', 'alphanumeric']
    difficulties = ['easy', 'medium', 'hard']
    
    for mode in modes:
        for difficulty in difficulties:
            print(f"Generating {mode} CAPTCHA with {difficulty} difficulty...")
            img_b64, text = generator.generate_captcha(mode=mode, difficulty=difficulty)
            print(f"Generated text: {text}")
            
            # Save example to file
            with open(f"example_{mode}_{difficulty}.html", "w") as f:
                f.write(f"""
                <!DOCTYPE html>
                <html>
                <head><title>CAPTCHA Example</title></head>
                <body>
                    <h2>{mode.capitalize()} CAPTCHA - {difficulty.capitalize()} Difficulty</h2>
                    <p>Text: {text}</p>
                    <img src="data:image/png;base64,{img_b64}" alt="CAPTCHA">
                </body>
                </html>
                """)
            
            print(f"Example saved to example_{mode}_{difficulty}.html")
            print("-" * 50)
    
    print("Testing complete!")