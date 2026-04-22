import numpy as np
import cv2
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import random
import base64
from io import BytesIO
import os

class MLCaptchaGenerator:
    def __init__(self, model_path="best_captcha_model.h5", img_width=200, img_height=60):
        self.img_width = img_width
        self.img_height = img_height
        self.model = None
        self.char_to_num = {str(i): i for i in range(10)}
        self.char_to_num[''] = 10
        self.num_to_char = {v: k for k, v in self.char_to_num.items()}
        
        # Load model if exists
        if os.path.exists(model_path):
            try:
                self.model = tf.keras.models.load_model(model_path)
                print("Captcha model loaded successfully!")
            except:
                print("Failed to load captcha model.")
        
        # Load fonts
        self.fonts = self._load_fonts()
        
    def _load_fonts(self):
        """Load available system fonts"""
        fonts = []
        font_sizes = [28, 32, 36, 40]
        
        font_names = [
            "arial.ttf", "arialbd.ttf", "times.ttf", "calibri.ttf", 
            "comic.ttf", "verdana.ttf", "tahoma.ttf"
        ]
        
        for font_name in font_names:
            for size in font_sizes:
                try:
                    font = ImageFont.truetype(font_name, size)
                    fonts.append(font)
                except:
                    pass
        
        if not fonts:
            for size in font_sizes:
                fonts.append(ImageFont.load_default())
        
        return fonts
    
    def generate_readable_captcha(self, text=None, difficulty="medium"):
        """Generate a readable captcha with controlled difficulty"""
        if text is None:
            text = ''.join(random.choices('0123456789', k=5))
        
        # Create image
        image = Image.new('RGB', (self.img_width, self.img_height), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        
        # Choose font
        font = random.choice(self.fonts)
        
        # Colors
        text_colors = [(0, 0, 0), (64, 64, 64), (128, 0, 0), (0, 128, 0), (0, 0, 128)]
        bg_noise_color = (220, 220, 220)
        
        # Add background noise based on difficulty
        if difficulty == "easy":
            noise_points = 30
            noise_lines = 2
        elif difficulty == "medium":
            noise_points = 80
            noise_lines = 4
        else:  # hard
            noise_points = 150
            noise_lines = 6
        
        # Add noise points
        for _ in range(noise_points):
            x = random.randint(0, self.img_width)
            y = random.randint(0, self.img_height)
            draw.point((x, y), fill=bg_noise_color)
        
        # Add noise lines
        for _ in range(noise_lines):
            x1 = random.randint(0, self.img_width)
            y1 = random.randint(0, self.img_height)
            x2 = random.randint(0, self.img_width)
            y2 = random.randint(0, self.img_height)
            draw.line([(x1, y1), (x2, y2)], fill=bg_noise_color, width=1)
        
        # Calculate text positioning
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Center text with slight randomization
        base_x = (self.img_width - text_width) // 2
        base_y = (self.img_height - text_height) // 2
        
        # Draw characters with slight variations
        char_width = text_width // len(text) if len(text) > 0 else 0
        
        for i, char in enumerate(text):
            # Position with controlled randomness
            if difficulty == "easy":
                x_offset = random.randint(-2, 2)
                y_offset = random.randint(-2, 2)
                angle = 0
            elif difficulty == "medium":
                x_offset = random.randint(-5, 5)
                y_offset = random.randint(-3, 3)
                angle = random.randint(-5, 5)
            else:  # hard
                x_offset = random.randint(-8, 8)
                y_offset = random.randint(-5, 5)
                angle = random.randint(-10, 10)
            
            char_x = base_x + i * char_width + x_offset
            char_y = base_y + y_offset
            color = random.choice(text_colors)
            
            if angle == 0:
                # No rotation
                draw.text((char_x, char_y), char, font=font, fill=color)
            else:
                # Create rotated character
                char_img = Image.new('RGBA', (50, 50), (0, 0, 0, 0))
                char_draw = ImageDraw.Draw(char_img)
                char_draw.text((25, 25), char, font=font, fill=color)
                
                rotated = char_img.rotate(angle, expand=1)
                image.paste(rotated, (char_x - 25, char_y - 25), rotated)
        
        return image, text
    
    def validate_captcha_difficulty(self, image, text):
        """Use ML model to validate if captcha is appropriately difficult"""
        if self.model is None:
            return True, 1.0  # No model available, assume valid
        
        try:
            # Preprocess image
            img_array = np.array(image.convert('L'))
            img_array = cv2.resize(img_array, (self.img_width, self.img_height))
            img_array = img_array.astype(np.float32) / 255.0
            img_array = img_array.reshape(1, self.img_height, self.img_width, 1)
            
            # Predict
            predictions = self.model.predict(img_array, verbose=0)
            
            # Decode prediction
            predicted_chars = []
            confidences = []
            
            for i in range(5):  # 5 characters
                pred_probs = predictions[i][0]
                pred_char = np.argmax(pred_probs)
                confidence = np.max(pred_probs)
                
                predicted_chars.append(pred_char)
                confidences.append(confidence)
            
            predicted_text = ''.join([str(char) if char < 10 else '' for char in predicted_chars])
            avg_confidence = np.mean(confidences)
            
            # Check if prediction matches actual text
            is_correct = predicted_text == text
            
            return is_correct, avg_confidence
            
        except Exception as e:
            print(f"Error in captcha validation: {e}")
            return True, 1.0
    
    def generate_adaptive_captcha(self, target_difficulty=0.7):
        """Generate captcha with adaptive difficulty based on ML model feedback"""
        max_attempts = 10
        attempt = 0
        
        while attempt < max_attempts:
            # Generate captcha with random difficulty
            difficulty_levels = ["easy", "medium", "hard"]
            difficulty = random.choice(difficulty_levels)
            
            image, text = self.generate_readable_captcha(difficulty=difficulty)
            
            # Validate difficulty
            is_correct, confidence = self.validate_captcha_difficulty(image, text)
            
            # Adjust based on target difficulty
            if target_difficulty <= 0.5:  # Easy captcha wanted
                if confidence > 0.8:  # Too easy
                    continue
            elif target_difficulty <= 0.8:  # Medium difficulty wanted
                if confidence < 0.4 or confidence > 0.9:
                    continue
            else:  # Hard captcha wanted
                if confidence > 0.6:
                    continue
            
            return image, text, confidence
            
        # Fallback: return medium difficulty
        return self.generate_readable_captcha(difficulty="medium") + (0.7,)
    
    def image_to_base64(self, image):
        """Convert PIL image to base64 string"""
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return img_str

# Integration with existing captcha system
def generate_ml_captcha(length=5, difficulty="medium"):
    """Generate ML-validated captcha for integration with Flask app"""
    generator = MLCaptchaGenerator()
    
    text = ''.join(random.choices('0123456789', k=length))
    image, actual_text, confidence = generator.generate_adaptive_captcha()
    
    return generator.image_to_base64(image), actual_text

if __name__ == "__main__":
    # Test the ML captcha generator
    generator = MLCaptchaGenerator()
    
    print("Generating test captchas...")
    
    for i in range(5):
        image, text, confidence = generator.generate_adaptive_captcha()
        image.save(f"test_captcha_{i}.png")
        print(f"Captcha {i+1}: {text} (confidence: {confidence:.3f})")
    
    print("Test captchas saved!")