"""
Digit Recognition Visualizer

This script helps visualize how your handwritten digits are processed and recognized
by the different models. It shows you the preprocessing steps and confidence levels,
which can help you understand why certain digits are being misclassified.

Instructions:
1. Run this script after capturing a few handwritten digits using the main app
2. Draw a digit and press Enter to see how it's processed and recognized
3. The visualization shows the raw image, preprocessed image, and predictions from 
   both the original and improved models
"""

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os

# Import our custom modules
from improved_number_recognition import preprocess_for_improved_recognition, predict_with_improved_model
from specialized_classifier_utils import enhance_air_writing_features, predict_with_specialized_model

# Load models
try:
    print("Loading digit recognition models...")
    original_model = load_model("bestmodel.h5")
    improved_model = load_model("distributed_training/model_outputs/simple_ensemble_model.h5")
    specialized_model = None
    if os.path.exists("5_vs_3_model.h5"):
        specialized_model = load_model("5_vs_3_model.h5")
        print("Specialized 5 vs 3 model loaded")
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)

# Set up camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit(1)

# Create canvas for drawing
canvas = np.zeros((480, 640, 3), np.uint8)
drawing = False
last_x, last_y = -1, -1
digit_label = ""

def mouse_callback(event, x, y, flags, param):
    global drawing, last_x, last_y, canvas
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        last_x, last_y = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(canvas, (last_x, last_y), (x, y), (255, 255, 255), 10)
            last_x, last_y = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

def preprocess_for_original_model(img):
    """Preprocess image for the original model"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(resized, (3, 3), 0)
    normalized = blurred.astype(np.float32) / 255.0
    return normalized, normalized.reshape(1, 28, 28, 1)

def get_original_model_prediction(img):
    """Get prediction from the original model"""
    _, preprocessed = preprocess_for_original_model(img)
    predictions = original_model.predict(preprocessed, verbose=0)[0]
    predicted_class = np.argmax(predictions)
    confidence = predictions[predicted_class]
    
    # Get top 3 predictions
    top_indices = np.argsort(predictions)[-3:][::-1]
    top_predictions = [(str(i), float(predictions[i])) for i in top_indices]
    
    return str(predicted_class), confidence, top_predictions

def get_improved_model_prediction(img):
    """Get prediction from the improved model"""
    digit, confidence = predict_with_improved_model(img)
    
    # For debugging, get the preprocessed image
    processed_img = preprocess_for_improved_recognition(img)
    
    # Get predictions for all classes for top-3
    predictions = improved_model.predict(processed_img, verbose=0)[0]
    top_indices = np.argsort(predictions)[-3:][::-1]
    top_predictions = [(str(i), float(predictions[i])) for i in top_indices]
    
    return digit, confidence, top_predictions, processed_img

def visualize_predictions(original_img, original_pred, improved_pred):
    """Visualize the predictions from both models"""
    plt.figure(figsize=(14, 7))
    
    # Original image
    plt.subplot(231)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title("Original Drawing")
    plt.axis('off')
    
    # Original model preprocessing
    orig_processed, _ = preprocess_for_original_model(original_img)
    plt.subplot(232)
    plt.imshow(orig_processed, cmap='gray')
    plt.title("Original Model Preprocessing")
    plt.axis('off')
    
    # Improved model preprocessing
    _, _, _, imp_processed = get_improved_model_prediction(original_img)
    plt.subplot(233)
    plt.imshow(imp_processed.reshape(28, 28), cmap='gray')
    plt.title("Improved Model Preprocessing")
    plt.axis('off')
    
    # Original model predictions
    plt.subplot(235)
    orig_digit, orig_conf, orig_top3 = original_pred
    plt.bar([p[0] for p in orig_top3], [p[1] for p in orig_top3], color='skyblue')
    plt.title(f"Original Model: {orig_digit} ({orig_conf:.2f})")
    plt.ylim(0, 1)
    
    # Improved model predictions
    plt.subplot(236)
    imp_digit, imp_conf, imp_top3, _ = improved_pred
    plt.bar([p[0] for p in imp_top3], [p[1] for p in imp_top3], color='lightgreen')
    plt.title(f"Improved Model: {imp_digit} ({imp_conf:.2f})")
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.show()

# Create window and set mouse callback
cv2.namedWindow("Digit Recognition Visualizer")
cv2.setMouseCallback("Digit Recognition Visualizer", mouse_callback)

print("Draw a digit in the window, then press Enter to analyze it")
print("Press 'c' to clear the canvas, 'q' to quit")

while True:
    # Display canvas
    display_img = canvas.copy()
    
    # Add instructions
    cv2.putText(display_img, "Draw a digit here", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
    cv2.putText(display_img, "Press 'Enter' to analyze, 'c' to clear, 'q' to quit", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    
    # Show the canvas
    cv2.imshow("Digit Recognition Visualizer", display_img)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == 13:  # Enter key
        if np.sum(canvas) > 1000:  # Make sure there's something drawn
            # Get predictions
            original_pred = get_original_model_prediction(canvas)
            improved_pred = get_improved_model_prediction(canvas)
            
            print("\nDigit Recognition Results:")
            print(f"Original model: {original_pred[0]} with confidence {original_pred[1]:.2f}")
            print(f"Improved model: {improved_pred[0]} with confidence {improved_pred[1]:.2f}")
            
            # Show visualization
            visualize_predictions(canvas, original_pred, improved_pred)
    
    elif key == ord('c'):  # Clear canvas
        canvas = np.zeros((480, 640, 3), np.uint8)
        print("Canvas cleared")
    
    elif key == ord('q'):  # Quit
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()