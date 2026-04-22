import random
import string
import math
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import base64

def generate_captcha_text(length=5):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

def generate_numeric_captcha(length=6):
    """Generate a numeric-only captcha with specified length"""
    return ''.join(random.choices(string.digits, k=length))

def generate_alpha_captcha(length=6, mode='auto'):
    """
    Generate an alphabetic-only captcha with specified length
    
    Args:
        length: Length of captcha
        mode: 'auto' (mixed case), 'uppercase', 'lowercase'
    """
    if mode == 'uppercase':
        letters = string.ascii_uppercase
    elif mode == 'lowercase':
        letters = string.ascii_lowercase
    else:  # auto or combined - use both cases
        letters = string.ascii_uppercase + string.ascii_lowercase
    
    return ''.join(random.choices(letters, k=length))

def generate_alphanumeric_captcha(length=6, mode='auto'):
    """
    Generate an alphanumeric captcha with specified length (letters and numbers)
    
    Args:
        length: Length of captcha
        mode: 'auto' (mixed case), 'uppercase', 'lowercase'
    """
    if mode == 'uppercase':
        letters = string.ascii_uppercase
    elif mode == 'lowercase':
        letters = string.ascii_lowercase
    else:  # auto or combined - use both cases
        letters = string.ascii_uppercase + string.ascii_lowercase
    
    digits = string.digits
    
    # Ensure at least 2 digits in the captcha
    num_digits = random.randint(2, max(2, length // 2))  # At least 2, at most half the length
    num_letters = length - num_digits
    
    # Generate the parts
    digit_part = ''.join(random.choices(digits, k=num_digits))
    letter_part = ''.join(random.choices(letters, k=num_letters))
    
    # Combine and shuffle
    combined = digit_part + letter_part
    combined_list = list(combined)
    random.shuffle(combined_list)
    
    return ''.join(combined_list)

def generate_captcha_image(text, style=None):
    """Generate captcha image with various modern styles"""
    if style is None:
        # Prefer readable styles for better user experience
        readable_styles = ['modern', 'gradient', 'shadow', 'neon', 'retro', 'sketch']
        style = random.choice(readable_styles)
    
    if style == 'modern':
        return _generate_modern_captcha(text)
    elif style == 'gradient':
        return _generate_gradient_captcha(text)
    elif style == 'wavy':
        return _generate_wavy_captcha(text)
    elif style == 'shadow':
        return _generate_shadow_captcha(text)
    elif style == 'neon':
        return _generate_neon_captcha(text)
    elif style == 'retro':
        return _generate_retro_captcha(text)
    elif style == 'matrix':
        return _generate_matrix_captcha(text)
    elif style == 'fire':
        return _generate_fire_captcha(text)
    elif style == 'ice':
        return _generate_ice_captcha(text)
    elif style == 'sketch':
        return _generate_sketch_captcha(text)
    elif style == 'neon_grid':
        return _generate_neon_grid_captcha(text)
    elif style == 'hologram':
        return _generate_hologram_captcha(text)
    elif style == 'laser':
        return _generate_laser_captcha(text)
    elif style == 'pixel':
        return _generate_pixel_captcha(text)
    elif style == 'rainbow':
        return _generate_rainbow_captcha(text)
    elif style == 'distorted':
        return _generate_distorted_captcha(text)
    elif style == 'overlapping':
        return _generate_overlapping_captcha(text)
    elif style == 'warped':
        return _generate_warped_captcha(text)
    else:
        return _generate_modern_captcha(text)

def _generate_modern_captcha(text):
    """Clean modern design with subtle effects"""
    width, height = 280, 100
    image = Image.new('RGB', (width, height), (248, 249, 250))
    
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except:
        font = ImageFont.load_default()
    
    draw = ImageDraw.Draw(image)
    
    # Add subtle background noise
    for _ in range(50):
        x = random.randint(0, width)
        y = random.randint(0, height)
        draw.point((x, y), fill=(220, 220, 220))
    
    # Draw each character with slight rotation and color variation
    colors = [(52, 58, 64), (73, 80, 87), (108, 117, 125)]
    
    # Calculate text positioning
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Center the text
    start_x = (width - text_width) // 2
    start_y = (height - text_height) // 2
    
    # Draw each character
    char_width = text_width // len(text) if len(text) > 0 else 0
    
    for i, char in enumerate(text):
        x = start_x + i * char_width + random.randint(-5, 5)
        y = start_y + random.randint(-3, 3)
        color = colors[i % len(colors)]
        
        # Simple rotation for variety
        angle = random.randint(-10, 10)
        
        if angle == 0:
            # No rotation needed
            draw.text((x, y), char, font=font, fill=color)
        else:
            # Create rotated character
            char_img = Image.new('RGBA', (60, 60), (0, 0, 0, 0))
            char_draw = ImageDraw.Draw(char_img)
            char_draw.text((30, 30), char, font=font, fill=color)
            
            # Rotate and paste
            rotated = char_img.rotate(angle, expand=1)
            
            # Calculate paste position
            paste_x = max(0, min(width - rotated.width, x - rotated.width // 2))
            paste_y = max(0, min(height - rotated.height, y - rotated.height // 2))
            
            # Ensure we're pasting RGBA image with proper alpha
            if rotated.mode == 'RGBA':
                image.paste(rotated, (paste_x, paste_y), rotated)
    
    # Add some decorative elements
    for _ in range(5):
        x1, y1 = random.randint(0, width), random.randint(0, height)
        x2, y2 = x1 + random.randint(2, 6), y1 + random.randint(2, 6)
        draw.rectangle([x1, y1, x2, y2], fill=(230, 230, 230))
    
    return _image_to_base64(image)

def _generate_gradient_captcha(text):
    """Gradient background with colorful text"""
    width, height = 280, 100
    image = Image.new('RGB', (width, height), (255, 255, 255))
    
    try:
        font = ImageFont.truetype("arial.ttf", 38)
    except:
        font = ImageFont.load_default()
    
    draw = ImageDraw.Draw(image)
    
    # Create gradient background
    for y in range(height):
        r = int(255 - (y / height) * 50)
        g = int(240 + (y / height) * 15)
        b = int(245 + (y / height) * 10)
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    
    # Draw text with rainbow colors
    gradient_colors = [(220, 53, 69), (255, 193, 7), (40, 167, 69), (0, 123, 255), (108, 117, 125), (220, 53, 69)]
    char_width = width // len(text)
    
    for i, char in enumerate(text):
        x = char_width * i + char_width // 3
        y = height // 3
        color = gradient_colors[i % len(gradient_colors)]
        
        # Add shadow effect
        draw.text((x + 2, y + 2), char, font=font, fill=(0, 0, 0, 100))
        draw.text((x, y), char, font=font, fill=color)
    
    # Add decorative circles
    for _ in range(15):
        x, y = random.randint(0, width), random.randint(0, height)
        r = random.randint(3, 8)
        color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
        draw.ellipse([x-r, y-r, x+r, y+r], fill=color)
    
    return _image_to_base64(image)

def _generate_wavy_captcha(text):
    """Wavy distorted text with smooth background"""
    width, height = 280, 100
    base_image = Image.new('RGB', (width, height), (240, 242, 247))
    
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()
    
    # Create text image
    text_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    text_draw = ImageDraw.Draw(text_img)
    
    # Center text
    bbox = text_draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    
    # Draw text with wavy effect
    for i, char in enumerate(text):
        char_x = x + i * (text_width // len(text))
        wave_y = y + math.sin(i * 0.8) * 8
        color = (random.randint(30, 80), random.randint(100, 150), random.randint(150, 200))
        text_draw.text((char_x, wave_y), char, font=font, fill=color)
    
    # Apply slight blur for smooth effect
    text_img = text_img.filter(ImageFilter.GaussianBlur(radius=0.5))
    base_image.paste(text_img, (0, 0), text_img)
    
    draw = ImageDraw.Draw(base_image)
    
    # Add wavy lines
    for _ in range(3):
        points = []
        for x in range(0, width, 10):
            y = height // 2 + math.sin(x * 0.02) * 20 + random.randint(-10, 10)
            points.append((x, y))
        if len(points) > 1:
            for i in range(len(points) - 1):
                draw.line([points[i], points[i + 1]], fill=(200, 200, 200), width=2)
    
    return _image_to_base64(base_image)

def _generate_shadow_captcha(text):
    """3D shadow effect with modern styling"""
    width, height = 280, 100
    image = Image.new('RGB', (width, height), (245, 246, 250))
    
    try:
        font = ImageFont.truetype("arial.ttf", 42)
    except:
        font = ImageFont.load_default()
    
    draw = ImageDraw.Draw(image)
    
    # Center text calculation
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    
    # Create 3D shadow effect
    shadow_colors = [(180, 180, 180), (160, 160, 160), (140, 140, 140)]
    
    # Draw multiple shadow layers
    for i, shadow_color in enumerate(shadow_colors):
        offset = i + 1
        draw.text((x + offset, y + offset), text, font=font, fill=shadow_color)
    
    # Draw main text
    main_color = (52, 58, 64)
    draw.text((x, y), text, font=font, fill=main_color)
    
    # Add geometric pattern background
    for _ in range(20):
        x1 = random.randint(0, width // 4)
        y1 = random.randint(0, height)
        x2 = x1 + random.randint(10, 30)
        y2 = y1 + random.randint(2, 6)
        draw.rectangle([x1, y1, x2, y2], fill=(230, 230, 230))
        
        x1 = random.randint(3 * width // 4, width)
        draw.rectangle([x1, y1, x1 + 20, y1 + 4], fill=(230, 230, 230))
    
    return _image_to_base64(image)

def _generate_neon_captcha(text):
    """Neon glow effect with dark background"""
    width, height = 280, 100
    image = Image.new('RGB', (width, height), (25, 30, 45))
    
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except:
        font = ImageFont.load_default()
    
    # Create glow effect by drawing text multiple times with different colors
    temp_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    temp_draw = ImageDraw.Draw(temp_img)
    
    # Center text
    bbox = temp_draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    
    # Neon colors
    neon_colors = [(0, 255, 255), (255, 20, 147), (50, 205, 50), (255, 165, 0), (138, 43, 226)]
    
    # Draw glow effect
    for i in range(len(text)):
        char_x = x + i * (text_width // len(text))
        char = text[i]
        color = neon_colors[i % len(neon_colors)]
        
        # Outer glow
        for offset in range(3, 0, -1):
            glow_color = tuple(int(c * (offset / 4)) for c in color)
            temp_draw.text((char_x - offset, y - offset), char, font=font, fill=glow_color + (100,))
            temp_draw.text((char_x + offset, y + offset), char, font=font, fill=glow_color + (100,))
        
        # Main text
        temp_draw.text((char_x, y), char, font=font, fill=color + (255,))
    
    # Apply blur for glow effect
    temp_img = temp_img.filter(ImageFilter.GaussianBlur(radius=1))
    image.paste(temp_img, (0, 0), temp_img)
    
    # Add subtle grid pattern
    draw = ImageDraw.Draw(image)
    for x_line in range(0, width, 40):
        draw.line([(x_line, 0), (x_line, height)], fill=(40, 45, 60), width=1)
    for y_line in range(0, height, 20):
        draw.line([(0, y_line), (width, y_line)], fill=(40, 45, 60), width=1)
    
    return _image_to_base64(image)

def _generate_retro_captcha(text):
    """Retro 80s style with scanlines and retro colors"""
    width, height = 280, 100
    image = Image.new('RGB', (width, height), (20, 20, 40))
    
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()
    
    draw = ImageDraw.Draw(image)
    
    # Retro color palette
    retro_colors = [(255, 20, 147), (0, 255, 255), (255, 165, 0), (50, 205, 50), (255, 69, 0)]
    
    # Center text
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    
    # Draw each character with retro styling
    for i, char in enumerate(text):
        char_x = x + i * (text_width // len(text))
        color = retro_colors[i % len(retro_colors)]
        
        # Draw with outline effect
        for dx in [-1, 1]:
            for dy in [-1, 1]:
                draw.text((char_x + dx, y + dy), char, font=font, fill=(0, 0, 0))
        draw.text((char_x, y), char, font=font, fill=color)
    
    # Add scanlines
    for y_line in range(0, height, 3):
        draw.line([(0, y_line), (width, y_line)], fill=(255, 255, 255, 30), width=1)
    
    return _image_to_base64(image)

def _generate_matrix_captcha(text):
    """Matrix-style green digital rain effect"""
    width, height = 280, 100
    image = Image.new('RGB', (width, height), (0, 0, 0))
    
    try:
        font = ImageFont.truetype("arial.ttf", 36)
        small_font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    draw = ImageDraw.Draw(image)
    
    # Add background matrix characters
    matrix_chars = "01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜｦﾝ"
    
    for _ in range(80):
        x = random.randint(0, width-10)
        y = random.randint(0, height-10)
        char = random.choice(matrix_chars)
        alpha = random.randint(20, 80)
        draw.text((x, y), char, font=small_font, fill=(0, 255, 0, alpha))
    
    # Center main text
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    
    # Draw main text with matrix green glow
    for i, char in enumerate(text):
        char_x = x + i * (text_width // len(text))
        
        # Glow effect
        for offset in range(2, 0, -1):
            glow_intensity = 100 + offset * 30
            draw.text((char_x - offset, y - offset), char, font=font, fill=(0, glow_intensity, 0))
            draw.text((char_x + offset, y + offset), char, font=font, fill=(0, glow_intensity, 0))
        
        # Main character
        draw.text((char_x, y), char, font=font, fill=(0, 255, 0))
    
    return _image_to_base64(image)

def _generate_fire_captcha(text):
    """Fire/flame effect with red-orange gradient"""
    width, height = 280, 100
    image = Image.new('RGB', (width, height), (40, 20, 0))
    
    try:
        font = ImageFont.truetype("arial.ttf", 38)
    except:
        font = ImageFont.load_default()
    
    draw = ImageDraw.Draw(image)
    
    # Fire colors
    fire_colors = [(255, 0, 0), (255, 69, 0), (255, 140, 0), (255, 165, 0), (255, 215, 0)]
    
    # Add flame particles in background
    for _ in range(50):
        x = random.randint(0, width)
        y = random.randint(height//2, height)
        size = random.randint(2, 6)
        color = random.choice(fire_colors)
        draw.ellipse([x, y, x+size, y+size], fill=color)
    
    # Center text
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    
    # Draw text with fire effect
    for i, char in enumerate(text):
        char_x = x + i * (text_width // len(text))
        
        # Multiple layers for fire effect
        for layer in range(3, 0, -1):
            color_idx = (i + layer) % len(fire_colors)
            color = fire_colors[color_idx]
            offset_y = random.randint(-layer, layer)
            draw.text((char_x, y + offset_y), char, font=font, fill=color)
    
    return _image_to_base64(image)

def _generate_ice_captcha(text):
    """Ice/frozen effect with blue-white crystals"""
    width, height = 280, 100
    image = Image.new('RGB', (width, height), (240, 248, 255))
    
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except:
        font = ImageFont.load_default()
    
    draw = ImageDraw.Draw(image)
    
    # Ice colors
    ice_colors = [(0, 191, 255), (135, 206, 235), (176, 224, 230), (173, 216, 230), (240, 248, 255)]
    
    # Add ice crystal effect
    for _ in range(30):
        x = random.randint(0, width)
        y = random.randint(0, height)
        size = random.randint(3, 8)
        color = random.choice(ice_colors)
        # Draw crystal shapes
        points = [(x, y-size), (x+size, y), (x, y+size), (x-size, y)]
        draw.polygon(points, fill=color, outline=(100, 149, 237))
    
    # Center text
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    
    # Draw text with ice effect
    for i, char in enumerate(text):
        char_x = x + i * (text_width // len(text))
        
        # Ice outline
        for dx in [-2, 0, 2]:
            for dy in [-2, 0, 2]:
                if dx != 0 or dy != 0:
                    draw.text((char_x + dx, y + dy), char, font=font, fill=(200, 230, 255))
        
        # Main character
        draw.text((char_x, y), char, font=font, fill=(0, 100, 200))
    
    return _image_to_base64(image)

def _generate_sketch_captcha(text):
    """Hand-drawn sketch style with pencil effect"""
    width, height = 280, 100
    image = Image.new('RGB', (width, height), (255, 255, 240))
    
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except:
        font = ImageFont.load_default()
    
    draw = ImageDraw.Draw(image)
    
    # Add paper texture
    for _ in range(200):
        x = random.randint(0, width)
        y = random.randint(0, height)
        draw.point((x, y), fill=(240, 240, 220))
    
    # Center text
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    
    # Draw text with sketch effect
    sketch_colors = [(70, 70, 70), (100, 100, 100), (50, 50, 50)]
    
    for i, char in enumerate(text):
        char_x = x + i * (text_width // len(text))
        color = sketch_colors[i % len(sketch_colors)]
        
        # Multiple overlapping drawings for sketch effect
        for offset in range(3):
            dx = random.randint(-1, 1)
            dy = random.randint(-1, 1)
            draw.text((char_x + dx, y + dy), char, font=font, fill=color)
    
    return _image_to_base64(image)

def _generate_neon_grid_captcha(text):
    """Cyberpunk neon grid with glowing text"""
    width, height = 280, 100
    image = Image.new('RGB', (width, height), (10, 10, 25))
    
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except:
        font = ImageFont.load_default()
    
    draw = ImageDraw.Draw(image)
    
    # Draw neon grid
    grid_color = (0, 255, 255)
    for x in range(0, width, 20):
        draw.line([(x, 0), (x, height)], fill=grid_color, width=1)
    for y in range(0, height, 15):
        draw.line([(0, y), (width, y)], fill=grid_color, width=1)
    
    # Center text
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    
    # Neon colors for each character
    neon_colors = [(255, 0, 255), (0, 255, 255), (255, 255, 0), (255, 0, 127), (0, 255, 127)]
    
    # Draw glowing text
    for i, char in enumerate(text):
        char_x = x + i * (text_width // len(text))
        color = neon_colors[i % len(neon_colors)]
        
        # Glow effect
        for radius in range(4, 0, -1):
            glow_color = tuple(int(c * (radius / 6)) for c in color)
            draw.text((char_x - radius//2, y - radius//2), char, font=font, fill=glow_color)
        
        # Main text
        draw.text((char_x, y), char, font=font, fill=color)
    
    return _image_to_base64(image)

def _generate_hologram_captcha(text):
    """Holographic effect with iridescent colors"""
    width, height = 280, 100
    image = Image.new('RGB', (width, height), (15, 15, 30))
    
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except:
        font = ImageFont.load_default()
    
    draw = ImageDraw.Draw(image)
    
    # Center text
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    
    # Holographic colors that shift
    holo_colors = [(255, 0, 255), (0, 255, 255), (255, 255, 0), (255, 127, 0), (127, 255, 0)]
    
    # Draw text with holographic effect
    for i, char in enumerate(text):
        char_x = x + i * (text_width // len(text))
        
        # Multiple colored layers for holographic effect
        for layer in range(5):
            color = holo_colors[(i + layer) % len(holo_colors)]
            alpha_color = tuple(int(c * 0.7) for c in color)
            offset_x = layer - 2
            draw.text((char_x + offset_x, y), char, font=font, fill=alpha_color)
    
    # Add holographic interference pattern
    for line_y in range(0, height, 4):
        for x_pos in range(0, width, 2):
            draw.point((x_pos, line_y), fill=(100, 100, 150))
    
    return _image_to_base64(image)

def _generate_laser_captcha(text):
    """Laser/sci-fi effect with beam-like text"""
    width, height = 280, 100
    image = Image.new('RGB', (width, height), (0, 0, 0))
    
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except:
        font = ImageFont.load_default()
    
    draw = ImageDraw.Draw(image)
    
    # Add laser beam background
    for i in range(10):
        y_pos = random.randint(0, height)
        draw.line([(0, y_pos), (width, y_pos)], fill=(255, 0, 0, 50), width=2)
    
    # Center text
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    
    # Laser colors
    laser_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    
    # Draw text with laser effect
    for i, char in enumerate(text):
        char_x = x + i * (text_width // len(text))
        color = laser_colors[i % len(laser_colors)]
        
        # Laser core
        draw.text((char_x, y), char, font=font, fill=color)
        
        # Laser glow
        for offset in range(1, 3):
            glow_color = tuple(int(c * 0.3) for c in color)
            draw.text((char_x - offset, y), char, font=font, fill=glow_color)
            draw.text((char_x + offset, y), char, font=font, fill=glow_color)
    
    return _image_to_base64(image)

def _generate_pixel_captcha(text):
    """Retro pixel art style with blocky text"""
    width, height = 280, 100
    image = Image.new('RGB', (width, height), (32, 32, 64))
    
    try:
        # Use a smaller font for pixel effect
        font = ImageFont.truetype("arial.ttf", 32)
    except:
        font = ImageFont.load_default()
    
    draw = ImageDraw.Draw(image)
    
    # Add pixel grid background
    pixel_size = 4
    for x in range(0, width, pixel_size * 4):
        for y in range(0, height, pixel_size * 4):
            if random.random() < 0.3:
                draw.rectangle([x, y, x + pixel_size, y + pixel_size], 
                             fill=(64, 64, 128))
    
    # Center text
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    
    # Pixel colors
    pixel_colors = [(255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 128, 0), (128, 255, 0)]
    
    # Draw pixelated text
    for i, char in enumerate(text):
        char_x = x + i * (text_width // len(text))
        color = pixel_colors[i % len(pixel_colors)]
        
        # Draw with pixel outline
        for dx in [-2, 0, 2]:
            for dy in [-2, 0, 2]:
                if abs(dx) + abs(dy) == 2:  # Only corners and sides
                    draw.text((char_x + dx, y + dy), char, font=font, fill=(0, 0, 0))
        
        draw.text((char_x, y), char, font=font, fill=color)
    
    return _image_to_base64(image)

def _generate_rainbow_captcha(text):
    """Rainbow gradient effect with colorful text"""
    width, height = 280, 100
    image = Image.new('RGB', (width, height), (255, 255, 255))
    
    try:
        font = ImageFont.truetype("arial.ttf", 38)
    except:
        font = ImageFont.load_default()
    
    draw = ImageDraw.Draw(image)
    
    # Rainbow colors
    rainbow_colors = [
        (255, 0, 0),    # Red
        (255, 127, 0),  # Orange
        (255, 255, 0),  # Yellow
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (75, 0, 130),   # Indigo
        (148, 0, 211)   # Violet
    ]
    
    # Center text
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    
    # Draw rainbow background arc
    for i in range(7):
        color = rainbow_colors[i]
        arc_y = height - 20 - i * 3
        draw.arc([(20, arc_y), (width-20, arc_y + 40)], 0, 180, fill=color, width=3)
    
    # Draw text with rainbow effect
    for i, char in enumerate(text):
        char_x = x + i * (text_width // len(text))
        color = rainbow_colors[i % len(rainbow_colors)]
        
        # Shadow
        draw.text((char_x + 2, y + 2), char, font=font, fill=(100, 100, 100))
        
        # Main character
        draw.text((char_x, y), char, font=font, fill=color)
    
    return _image_to_base64(image)

def _generate_distorted_captcha(text):
    """Advanced distorted captcha with overlapping and warping"""
    width, height = 350, 120
    image = Image.new('RGB', (width, height), (240, 240, 240))
    
    try:
        fonts = [
            ImageFont.truetype("arial.ttf", 48),
            ImageFont.truetype("arialbd.ttf", 52),
            ImageFont.truetype("times.ttf", 46),
            ImageFont.truetype("comic.ttf", 50),
        ]
    except:
        fonts = [ImageFont.load_default()]
    
    draw = ImageDraw.Draw(image)
    
    # Add complex background noise
    for _ in range(200):
        x = random.randint(0, width)
        y = random.randint(0, height)
        size = random.randint(1, 3)
        color = (random.randint(180, 220), random.randint(180, 220), random.randint(180, 220))
        draw.ellipse([x, y, x+size, y+size], fill=color)
    
    # Add interference lines
    for _ in range(12):
        x1, y1 = random.randint(0, width), random.randint(0, height)
        x2, y2 = random.randint(0, width), random.randint(0, height)
        color = (random.randint(150, 200), random.randint(150, 200), random.randint(150, 200))
        draw.line([(x1, y1), (x2, y2)], fill=color, width=random.randint(1, 2))
    
    # Character positioning with significant overlap
    char_positions = []
    base_width = (width - 80) / len(text)
    
    for i in range(len(text)):
        # Allow substantial overlap
        overlap_factor = random.uniform(-0.4, 0.3)
        x = 40 + base_width * i + (base_width * overlap_factor)
        y = height // 2 + random.randint(-20, 20)
        char_positions.append((x, y))
    
    # Draw characters with various distortions
    colors = [(20, 20, 20), (60, 60, 60), (100, 50, 50), (50, 100, 50), (50, 50, 100)]
    
    for i, (char, (x, y)) in enumerate(zip(text, char_positions)):
        # Random font and size variation
        font = random.choice(fonts)
        
        # Create character on temporary image for transformations
        temp_size = 120
        temp_img = Image.new('RGBA', (temp_size, temp_size), (0, 0, 0, 0))
        temp_draw = ImageDraw.Draw(temp_img)
        
        # Character color
        color = colors[i % len(colors)]
        
        # Add shadow/outline for complexity
        for offset in range(3, 0, -1):
            shadow_color = tuple(c + offset * 20 for c in color)
            shadow_color = tuple(min(255, c) for c in shadow_color)
            temp_draw.text((temp_size//2 + offset, temp_size//2 + offset), char, 
                          font=font, fill=shadow_color)
        
        # Main character
        temp_draw.text((temp_size//2, temp_size//2), char, font=font, fill=color)
        
        # Apply random transformations
        
        # 1. Random rotation
        angle = random.randint(-35, 35)
        temp_img = temp_img.rotate(angle, expand=1)
        
        # 2. Random scaling
        scale_x = random.uniform(0.7, 1.3)
        scale_y = random.uniform(0.8, 1.2)
        new_size = (int(temp_img.width * scale_x), int(temp_img.height * scale_y))
        temp_img = temp_img.resize(new_size, Image.Resampling.LANCZOS)
        
        # 3. Shearing effect (simple skew)
        if random.random() < 0.5:
            # Create skewed version
            skew_factor = random.uniform(-0.3, 0.3)
            skewed_img = Image.new('RGBA', (temp_img.width + 40, temp_img.height), (0, 0, 0, 0))
            for py in range(temp_img.height):
                offset_x = int(py * skew_factor)
                for px in range(temp_img.width):
                    if 0 <= px + offset_x < skewed_img.width:
                        try:
                            pixel = temp_img.getpixel((px, py))
                            skewed_img.putpixel((px + offset_x + 20, py), pixel)
                        except:
                            pass
            temp_img = skewed_img
        
        # Paste character onto main image
        paste_x = int(x - temp_img.width // 2)
        paste_y = int(y - temp_img.height // 2)
        
        # Ensure paste position is within bounds
        paste_x = max(0, min(width - temp_img.width, paste_x))
        paste_y = max(0, min(height - temp_img.height, paste_y))
        
        image.paste(temp_img, (paste_x, paste_y), temp_img)
    
    # Add final noise layer
    noise_overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    noise_draw = ImageDraw.Draw(noise_overlay)
    
    for _ in range(50):
        x = random.randint(0, width)
        y = random.randint(0, height)
        size = random.randint(1, 2)
        alpha = random.randint(30, 100)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), alpha)
        noise_draw.ellipse([x, y, x+size, y+size], fill=color)
    
    image = Image.alpha_composite(image.convert('RGBA'), noise_overlay).convert('RGB')
    
    return _image_to_base64(image)

def _generate_overlapping_captcha(text):
    """Captcha with intentionally overlapping characters"""
    width, height = 350, 120
    
    # Create base image with texture
    image = Image.new('RGB', (width, height), (245, 245, 245))
    
    # Add subtle texture
    for _ in range(300):
        x = random.randint(0, width)
        y = random.randint(0, height)
        color = (random.randint(235, 255), random.randint(235, 255), random.randint(235, 255))
        image.putpixel((x, y), color)
    
    draw = ImageDraw.Draw(image)
    
    try:
        fonts = [
            ImageFont.truetype("arial.ttf", 55),
            ImageFont.truetype("arialbd.ttf", 58),
            ImageFont.truetype("times.ttf", 52),
        ]
    except:
        fonts = [ImageFont.load_default()]
    
    # Intentionally create overlapping layout
    positions = []
    base_spacing = (width - 100) / (len(text) - 1) if len(text) > 1 else 0
    
    for i in range(len(text)):
        if i == 0:
            x = 50
        else:
            # Reduce spacing to create overlap
            x = positions[-1][0] + base_spacing * random.uniform(0.4, 0.8)
        
        y = height // 2 + random.randint(-15, 15)
        positions.append((x, y))
    
    # Draw characters with overlapping
    colors = [(40, 40, 40), (80, 40, 40), (40, 80, 40), (40, 40, 80), (80, 40, 80)]
    
    for i, (char, (x, y)) in enumerate(zip(text, positions)):
        font = random.choice(fonts)
        color = colors[i % len(colors)]
        
        # Create character with effects
        char_img = Image.new('RGBA', (100, 100), (0, 0, 0, 0))
        char_draw = ImageDraw.Draw(char_img)
        
        # Add outline for better separation when overlapping
        for dx in [-2, -1, 0, 1, 2]:
            for dy in [-2, -1, 0, 1, 2]:
                if dx != 0 or dy != 0:
                    outline_color = tuple(min(255, c + 100) for c in color)
                    char_draw.text((50 + dx, 50 + dy), char, font=font, fill=outline_color)
        
        # Main character
        char_draw.text((50, 50), char, font=font, fill=color)
        
        # Random rotation
        angle = random.randint(-20, 20)
        char_img = char_img.rotate(angle, expand=1)
        
        # Paste with overlap
        paste_x = int(x - char_img.width // 2)
        paste_y = int(y - char_img.height // 2)
        
        # Use alpha blending for overlapping effect
        if char_img.mode == 'RGBA':
            image.paste(char_img, (paste_x, paste_y), char_img)
    
    # Add distraction elements
    for _ in range(15):
        x = random.randint(0, width)
        y = random.randint(0, height)
        size = random.randint(3, 8)
        color = (random.randint(200, 230), random.randint(200, 230), random.randint(200, 230))
        draw.ellipse([x, y, x+size, y+size], fill=color)
    
    return _image_to_base64(image)

def _generate_warped_captcha(text):
    """Captcha with wave distortions and warping"""
    width, height = 350, 120
    
    # Create larger temporary image for warping
    temp_width, temp_height = width + 100, height + 50
    temp_img = Image.new('RGB', (temp_width, temp_height), (250, 250, 250))
    temp_draw = ImageDraw.Draw(temp_img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 60)
    except:
        font = ImageFont.load_default()
    
    # Draw text normally first
    bbox = temp_draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (temp_width - text_width) // 2
    y = (temp_height - text_height) // 2
    
    # Add background pattern
    for i in range(0, temp_width, 20):
        for j in range(0, temp_height, 20):
            if random.random() < 0.1:
                temp_draw.rectangle([i, j, i+2, j+2], fill=(230, 230, 230))
    
    # Draw each character with individual effects
    char_width = text_width // len(text) if len(text) > 0 else 0
    colors = [(60, 60, 60), (100, 60, 60), (60, 100, 60), (60, 60, 100)]
    
    for i, char in enumerate(text):
        char_x = x + i * char_width + random.randint(-10, 10)
        char_y = y + random.randint(-15, 15)
        color = colors[i % len(colors)]
        
        # Add shadow
        temp_draw.text((char_x + 3, char_y + 3), char, font=font, fill=(180, 180, 180))
        temp_draw.text((char_x, char_y), char, font=font, fill=color)
    
    # Apply wave distortion
    distorted = Image.new('RGB', (width, height), (250, 250, 250))
    
    for y in range(height):
        for x in range(width):
            # Calculate source coordinates with wave distortion
            wave_x = x + int(15 * math.sin(y * 0.1))
            wave_y = y + int(8 * math.sin(x * 0.08))
            
            # Add bounds checking
            src_x = min(temp_width - 1, max(0, wave_x + 50))
            src_y = min(temp_height - 1, max(0, wave_y + 25))
            
            try:
                pixel = temp_img.getpixel((src_x, src_y))
                distorted.putpixel((x, y), pixel)
            except:
                pass
    
    # Add noise
    draw = ImageDraw.Draw(distorted)
    for _ in range(100):
        x = random.randint(0, width)
        y = random.randint(0, height)
        draw.point((x, y), fill=(random.randint(200, 255), random.randint(200, 255), random.randint(200, 255)))
    
    return _image_to_base64(distorted)

def _image_to_base64(image):
    """Convert PIL image to base64 string"""
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_str
