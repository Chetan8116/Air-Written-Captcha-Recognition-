import os
import random
import string
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps, ImageEnhance
import cv2
import json
from io import BytesIO
import base64
from tqdm import tqdm
import matplotlib.pyplot as plt

class EnhancedCaptchaGenerator:
    def __init__(self, dataset_path="enhanced_numeric_captcha_dataset", img_width=200, img_height=80):
        self.dataset_path = dataset_path
        self.img_width = img_width
        self.img_height = img_height
        self.fonts = self._load_fonts()
        
        # More realistic colors - Cloudflare-like colors
        self.text_colors = [
            (66, 66, 66), (77, 77, 77), (89, 89, 89),   # Dark gray variants
            (54, 73, 98), (60, 82, 106), (72, 95, 121), # Blue-gray variants
            (70, 70, 85), (80, 80, 95), (90, 90, 105),  # Slate gray variants
            (76, 66, 66), (86, 76, 76), (96, 86, 86)    # Brown-gray variants
        ]
        
        # Cloudflare-like background colors (subtle variations of off-white)
        self.bg_colors = [
            (248, 250, 252), (250, 251, 253), (246, 248, 250),
            (245, 247, 249), (249, 249, 251), (247, 249, 250)
        ]
        
        # Create directories
        os.makedirs(os.path.join(dataset_path, "images"), exist_ok=True)
        os.makedirs(os.path.join(dataset_path, "labels"), exist_ok=True)
        
    def _load_fonts(self):
        """Load a variety of fonts that work well for captchas"""
        fonts = []
        font_sizes = [30, 32, 34, 36, 38, 40]
        
        # Common fonts found in captchas - prioritize fonts that look more like typical captchas
        font_names = [
            "arial.ttf", "arialbd.ttf", "ariblk.ttf",
            "times.ttf", "timesbd.ttf", "timesi.ttf",
            "verdana.ttf", "verdanab.ttf",
            "tahoma.ttf", "tahomabd.ttf",
            "segoeui.ttf", "segoeuib.ttf",
            "consola.ttf", "consolab.ttf",
            "lucon.ttf", "micross.ttf",
            "trebuc.ttf", "trebucbd.ttf"
        ]
        
        # Windows font directories
        font_dirs = [
            "C:\\Windows\\Fonts",
            "C:\\Windows\\winsxs\\amd64_microsoft-windows-font-truetype-arial_31bf3856ad364e35_6.1.7600.16385_none_30e30305a93daf1e"
        ]
        
        # Try to load fonts from Windows directories first
        for font_dir in font_dirs:
            if os.path.exists(font_dir):
                for font_name in font_names:
                    font_path = os.path.join(font_dir, font_name)
                    if os.path.exists(font_path):
                        for size in font_sizes:
                            try:
                                font = ImageFont.truetype(font_path, size)
                                fonts.append(font)
                            except:
                                pass
        
        # If no fonts were found, try direct loading
        if not fonts:
            for font_name in font_names:
                for size in font_sizes:
                    try:
                        font = ImageFont.truetype(font_name, size)
                        fonts.append(font)
                    except:
                        pass
        
        # Fallback to default font if still no fonts
        if not fonts:
            for size in font_sizes:
                fonts.append(ImageFont.load_default())
        
        return fonts
    
    def generate_captcha_text(self, length=6):
        """Generate random numeric captcha text"""
        return ''.join(random.choices(string.digits, k=length))
        
    def _add_noise_dots(self, image, draw, density="medium"):
        """Add noise dots with specified density"""
        if density == "low":
            num_dots = random.randint(30, 60)
            dot_size_range = (1, 2)
        elif density == "medium":
            num_dots = random.randint(60, 150)
            dot_size_range = (1, 2)
        else:  # high
            num_dots = random.randint(120, 250)
            dot_size_range = (1, 3)
            
        for _ in range(num_dots):
            x = random.randint(0, self.img_width)
            y = random.randint(0, self.img_height)
            size = random.randint(*dot_size_range)
            # Use a lighter color for dots to mimic Cloudflare style
            brightness = random.randint(180, 230)
            color = (brightness, brightness, brightness)
            draw.ellipse((x, y, x+size, y+size), fill=color)
        
    def _add_noise_lines(self, image, draw, density="medium"):
        """Add noise lines with specified density"""
        if density == "low":
            num_lines = random.randint(2, 4)
            width_range = (1, 1)
        elif density == "medium":
            num_lines = random.randint(3, 6)
            width_range = (1, 2)
        else:  # high
            num_lines = random.randint(5, 8)
            width_range = (1, 2)
        
        # Cloudflare-like noise lines (subtle gray tones)
        line_colors = [
            (210, 210, 210), (200, 200, 200), (190, 190, 190),
            (215, 215, 220), (205, 205, 210), (195, 195, 200)
        ]
        
        for _ in range(num_lines):
            # Create wavy lines
            start_x = random.randint(0, self.img_width // 4)
            end_x = random.randint(3 * self.img_width // 4, self.img_width)
            
            # Create lines that cross through the text area
            start_y = random.randint(self.img_height // 4, 3 * self.img_height // 4)
            end_y = random.randint(self.img_height // 4, 3 * self.img_height // 4)
            
            # Cloudflare-style lines often go more horizontally than vertically
            width = random.randint(*width_range)
            color = random.choice(line_colors)
            
            # Create wavy line with multiple segments
            points = []
            segments = random.randint(3, 6)
            
            for i in range(segments + 1):
                x = start_x + (end_x - start_x) * i / segments
                # Add some waviness
                y = start_y + (end_y - start_y) * i / segments
                y += random.randint(-10, 10)
                y = max(5, min(self.img_height - 5, y))  # Keep in bounds
                
                points.append((x, y))
            
            # Draw line segments
            for i in range(len(points) - 1):
                draw.line([points[i], points[i + 1]], fill=color, width=width)
    
    def _add_grid_pattern(self, image, draw):
        """Add subtle grid pattern occasionally"""
        if random.random() < 0.3:  # 30% chance
            grid_color = (230, 230, 230)
            grid_spacing = random.randint(15, 25)
            
            # Draw vertical lines
            for x in range(0, self.img_width, grid_spacing):
                draw.line([(x, 0), (x, self.img_height)], fill=grid_color, width=1)
            
            # Draw horizontal lines
            for y in range(0, self.img_height, grid_spacing):
                draw.line([(0, y), (self.img_width, y)], fill=grid_color, width=1)
    
    def _apply_cloudflare_style_distortion(self, image):
        """Apply Cloudflare-like distortions to the image"""
        img_array = np.array(image)
        
        # Apply slight wave distortion (vertical and horizontal)
        h, w = img_array.shape[:2]
        
        # Create distortion map
        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)
        
        # Cloudflare-like wave pattern (subtle)
        wave_amplitude_x = random.uniform(2.0, 4.0)
        wave_amplitude_y = random.uniform(1.0, 3.0)
        wave_frequency_x = random.uniform(0.05, 0.1)
        wave_frequency_y = random.uniform(0.05, 0.1)
        
        # Create distortion maps
        for y in range(h):
            for x in range(w):
                # Subtle sine wave distortion
                map_x[y, x] = x + wave_amplitude_x * np.sin(y * wave_frequency_y)
                map_y[y, x] = y + wave_amplitude_y * np.sin(x * wave_frequency_x)
        
        # Apply distortion
        distorted = cv2.remap(img_array, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        
        # Convert back to PIL
        return Image.fromarray(distorted)
    
    def _apply_subtle_blur(self, image):
        """Apply very subtle blur to smooth edges like real captchas"""
        if random.random() < 0.4:  # 40% chance
            radius = random.uniform(0.3, 0.7)  # Very subtle blur
            return image.filter(ImageFilter.GaussianBlur(radius=radius))
        return image
    
    def _adjust_contrast_and_sharpness(self, image):
        """Adjust contrast and sharpness to match Cloudflare captchas"""
        # Slightly increase contrast
        contrast_enhancer = ImageEnhance.Contrast(image)
        contrast_factor = random.uniform(1.1, 1.3)  # Subtle increase
        image = contrast_enhancer.enhance(contrast_factor)
        
        # Slightly adjust sharpness
        sharpness_enhancer = ImageEnhance.Sharpness(image)
        sharpness_factor = random.uniform(0.8, 1.2)
        image = sharpness_enhancer.enhance(sharpness_factor)
        
        return image
    
    def generate_single_captcha(self, text=None, difficulty="medium"):
        """Generate a single captcha image with label"""
        if text is None:
            text = self.generate_captcha_text()
        
        # Create image with random background color
        bg_color = random.choice(self.bg_colors)
        image = Image.new('RGB', (self.img_width, self.img_height), bg_color)
        draw = ImageDraw.Draw(image)
        
        # Add subtle background pattern occasionally
        self._add_grid_pattern(image, draw)
        
        # Choose font
        font = random.choice(self.fonts)
        
        # Calculate text positioning for better centering
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Center text with slight randomization
        base_x = (self.img_width - text_width) // 2 + random.randint(-10, 10)
        base_y = (self.img_height - text_height) // 2 + random.randint(-5, 5)
        
        # Ensure text stays within bounds
        base_x = max(5, min(base_x, self.img_width - text_width - 5))
        base_y = max(5, min(base_y, self.img_height - text_height - 5))
        
        # Draw each character with individual styling - Cloudflare style
        avg_char_width = text_width / len(text)
        
        for i, char in enumerate(text):
            # Calculate position with slight variation between characters
            if i == 0:
                char_x = base_x
            else:
                # Variable spacing between characters
                spacing_factor = random.uniform(0.85, 1.15)
                char_x += int(avg_char_width * spacing_factor)
            
            # Vertical variation
            char_y = base_y + random.randint(-3, 3)
            
            # Random rotation for each character (Cloudflare-like subtle rotations)
            angle = random.uniform(-8, 8)
            
            # Random color for each character from palette
            color = random.choice(self.text_colors)
            
            if angle == 0:
                # No rotation needed
                draw.text((char_x, char_y), char, font=font, fill=color)
            else:
                # Create character on transparent background
                char_img = Image.new('RGBA', (int(avg_char_width * 1.5), int(text_height * 1.5)), (0, 0, 0, 0))
                char_draw = ImageDraw.Draw(char_img)
                
                # Draw centered
                char_bbox = char_draw.textbbox((0, 0), char, font=font)
                char_text_width = char_bbox[2] - char_bbox[0]
                char_text_height = char_bbox[3] - char_bbox[1]
                
                # Center in the temp image
                cx = (char_img.width - char_text_width) // 2
                cy = (char_img.height - char_text_height) // 2
                
                char_draw.text((cx, cy), char, font=font, fill=color)
                
                # Rotate and paste
                rotated = char_img.rotate(angle, expand=0, resample=Image.BICUBIC)
                
                # Paste with transparency
                image.paste(rotated, (int(char_x - avg_char_width * 0.25), int(char_y - text_height * 0.25)), rotated)
        
        # Add noise elements based on difficulty
        self._add_noise_dots(image, draw, difficulty)
        self._add_noise_lines(image, draw, difficulty)
        
        # Apply Cloudflare-like distortion
        image = self._apply_cloudflare_style_distortion(image)
        
        # Apply subtle blur
        image = self._apply_subtle_blur(image)
        
        # Adjust contrast and sharpness
        image = self._adjust_contrast_and_sharpness(image)
        
        return image, text
    
    def generate_dataset(self, num_samples=10000, difficulty="medium"):
        """Generate complete dataset with images and labels"""
        print(f"Generating {num_samples} enhanced numeric captcha samples...")
        
        labels = []
        
        for i in tqdm(range(num_samples)):
            # Generate captcha with specified difficulty
            image, text = self.generate_single_captcha(difficulty=difficulty)
            
            # Save image
            image_filename = f"captcha_{i:06d}.png"
            image_path = os.path.join(self.dataset_path, "images", image_filename)
            image.save(image_path)
            
            # Store label information
            labels.append({
                "filename": image_filename,
                "text": text,
                "length": len(text)
            })
        
        # Save labels file
        labels_path = os.path.join(self.dataset_path, "labels.json")
        with open(labels_path, 'w') as f:
            json.dump(labels, f, indent=2)
        
        print(f"Dataset generated successfully!")
        print(f"Images saved to: {os.path.join(self.dataset_path, 'images')}")
        print(f"Labels saved to: {labels_path}")
        
        return labels
    
    def preview_samples(self, num_samples=10, difficulty="medium"):
        """Generate and display sample captchas"""
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()
        
        for i in range(num_samples):
            image, text = self.generate_single_captcha(difficulty=difficulty)
            axes[i].imshow(image)
            axes[i].set_title(f"Text: {text}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return True
    
    def generate_captcha_b64(self, length=6, difficulty="medium", use_distortion=False, use_perspective=False, use_noise=False):
        """Generate a captcha and return base64 string and text"""
        # Generate text if needed based on length
        text = None
        if length > 0:
            text = ''.join(random.choice("0123456789") for _ in range(length))
            
        image, text = self.generate_single_captcha(text=text, difficulty=difficulty)
        
        # Convert to base64
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return img_str, text

if __name__ == "__main__":
    # Initialize generator
    generator = EnhancedCaptchaGenerator()
    
    print("Enhanced Numeric Captcha Generator")
    print("=" * 50)
    
    # Preview some samples first
    print("Generating preview samples...")
    generator.preview_samples(difficulty="medium")
    
    # Ask user for dataset size
    while True:
        try:
            num_samples = int(input("\nEnter number of samples to generate (recommended: 5000-10000): "))
            if num_samples > 0:
                break
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Generate dataset
    labels = generator.generate_dataset(num_samples, difficulty="medium")
    
    print(f"\nDataset Statistics:")
    print(f"Total samples: {len(labels)}")
    print(f"Sample text lengths: {set(len(label['text']) for label in labels)}")