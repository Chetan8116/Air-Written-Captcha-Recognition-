"""
Special B Recognition Test
This script focuses specifically on improving the recognition of the letter 'B'
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from new_alphabet_integration import load_alphabet_model, predict_alphabet

# Ensure output directory exists
os.makedirs('debug_output', exist_ok=True)

# Create a specialized 'B' image that mimics the sample in your attachment
img = np.zeros((200, 200), dtype=np.uint8)

# Draw a more accurate B shape
# Vertical line
cv2.line(img, (50, 40), (50, 160), 255, 15)
# Top curve
cv2.ellipse(img, (70, 70), (35, 30), 0, 270, 90, 255, 15)
# Bottom curve
cv2.ellipse(img, (70, 130), (35, 30), 0, 270, 90, 255, 15)

# Save the test image
cv2.imwrite('debug_output/test_b_accurate.png', img)

# Load the model
load_alphabet_model()

# Test the model with the new image
predicted_char, confidence, results = predict_alphabet(img, uppercase_only=True)

print(f"Prediction for B: {predicted_char} with confidence {confidence:.4f}")
print("\nTop 5 predictions:")
top_chars = results.get("top_chars", [])
top_confidences = results.get("top_confidences", [])
for i, (char, conf) in enumerate(zip(top_chars, top_confidences)):
    print(f"{i+1}. {char} - {conf:.4f}")

# Now let's add a specialized B detector based on structural analysis
def detect_letter_b(image):
    """
    Specialized detector for the letter B based on structural analysis
    """
    # Convert to grayscale if needed
    if len(image.shape) > 2 and image.shape[2] > 1:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Make sure image is uint8
    if np.max(gray) <= 1.0:
        gray = (gray * 255).astype(np.uint8)
    
    # Apply threshold
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return False, 0.0
    
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Create a mask of just the letter
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], 0, 255, -1)
    
    # Divide the mask into vertical sections
    h, w = mask.shape
    left_edge = mask[:, :int(w*0.25)]
    middle = mask[:, int(w*0.25):int(w*0.75)]
    right_edge = mask[:, int(w*0.75):]
    
    # B typically has a strong vertical line on the left
    left_edge_filled = np.sum(left_edge > 0) / (h * w * 0.25)
    
    # B typically has two bulges in the middle section (top and bottom curves)
    middle_filled = np.sum(middle > 0) / (h * w * 0.5)
    
    # B typically has less on the right edge
    right_edge_filled = np.sum(right_edge > 0) / (h * w * 0.25)
    
    # Calculate vertical division to check for two loops
    top_half = mask[:int(h*0.5), :]
    bottom_half = mask[int(h*0.5):, :]
    
    top_filled = np.sum(top_half > 0) / (h * w * 0.5)
    bottom_filled = np.sum(bottom_half > 0) / (h * w * 0.5)
    
    # B has a good balance between top and bottom halves
    balanced_halves = abs(top_filled - bottom_filled) < 0.2
    
    # B has strong left edge and moderate middle with less right edge
    b_like_structure = (left_edge_filled > 0.6) and (middle_filled > 0.3) and (right_edge_filled < left_edge_filled)
    
    # Calculate a confidence score for being a B
    b_score = 0.0
    
    if b_like_structure:
        b_score += 0.5
    
    if balanced_halves:
        b_score += 0.3
    
    # Check the aspect ratio (B is typically taller than wide)
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = w / h if h > 0 else 0
    
    if 0.5 <= aspect_ratio <= 0.8:  # Typical B aspect ratio
        b_score += 0.2
    
    # Limit the score to 0.95
    b_score = min(b_score, 0.95)
    
    return b_score > 0.5, b_score

# Apply our specialized B detector
is_b, b_confidence = detect_letter_b(img)
print(f"\nSpecialized B detector: {'B' if is_b else 'Not B'} with confidence {b_confidence:.4f}")

# Create a visualization
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("B Test Image")
plt.imshow(img, cmap='gray')
plt.axis('off')

# Create a combined result showing both methods
combined_result = f"Model: {predicted_char} ({confidence:.2f})\nSpecialized: {'B' if is_b else 'Not B'} ({b_confidence:.2f})"
plt.subplot(1, 2, 2)
plt.title(combined_result)
# Show the image with edges highlighted to show what features were detected
edges = cv2.Canny(img, 100, 200)
plt.imshow(edges, cmap='viridis')
plt.axis('off')

plt.tight_layout()
plt.savefig('debug_output/b_detection_specialized.png')

print("\nResults saved to debug_output/b_detection_specialized.png")