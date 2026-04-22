"""
Visualize letter detection performance with selected test images
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import glob

def visualize_detection_performance():
    """
    Create a visualization of the letter detection performance
    """
    # Check if metrics file exists
    if not os.path.exists("test_results/letter_detection_metrics.json"):
        print("Metrics file not found. Run test_all_letter_detectors.py first.")
        return
        
    # Load metrics
    with open("test_results/letter_detection_metrics.json", "r") as f:
        metrics = json.load(f)
    
    # Get overall metrics
    overall = metrics["overall"]
    
    # Create a bar chart of accuracy for each letter
    letters = []
    accuracies = []
    
    for letter in sorted(metrics["per_letter"].keys()):
        # Skip letters with no test data
        if "true_positive" not in metrics["per_letter"][letter]:
            continue
            
        tp = metrics["per_letter"][letter]["true_positive"]
        tn = metrics["per_letter"][letter]["true_negative"]
        fp = metrics["per_letter"][letter]["false_positive"]
        fn = metrics["per_letter"][letter]["false_negative"]
        
        if tp + tn + fp + fn == 0:
            continue
            
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        letters.append(letter)
        accuracies.append(accuracy)
    
    # Create the visualization
    plt.figure(figsize=(15, 10))
    
    # Plot the accuracy bar chart
    plt.subplot(2, 1, 1)
    plt.bar(letters, accuracies, color='skyblue')
    plt.axhline(y=overall["accuracy"], color='r', linestyle='-', label=f'Average: {overall["accuracy"]:.2f}')
    plt.ylim(0, 1.0)
    plt.title('Letter Detection Accuracy by Letter')
    plt.xlabel('Letter')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot 3 example images
    test_files = glob.glob("test_images/*.png")
    plt.subplot(2, 3, 4)
    
    # Try to find a successful B detection
    b_files = [f for f in test_files if os.path.basename(f).startswith("B_")]
    if b_files:
        img = cv2.imread(b_files[0])
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb)
            plt.title("Example B")
            plt.axis('off')
    
    # Try to find a successful Z detection (since it has perfect accuracy)
    plt.subplot(2, 3, 5)
    z_files = [f for f in test_files if os.path.basename(f).startswith("Z_")]
    if z_files:
        img = cv2.imread(z_files[0])
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb)
            plt.title("Example Z")
            plt.axis('off')
    
    # Try to find a successful W detection
    plt.subplot(2, 3, 6)
    w_files = [f for f in test_files if os.path.basename(f).startswith("W_")]
    if w_files:
        img = cv2.imread(w_files[0])
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb)
            plt.title("Example W")
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("test_results/letter_detection_visualization.png")
    print("Visualization saved to test_results/letter_detection_visualization.png")
    
    # Also create a confusion matrix visualization
    plt.figure(figsize=(15, 12))
    
    # Create a pseudo-confusion matrix showing detection rate
    detected = np.zeros((26, 26))
    
    # Check which letters are available in the metrics
    available_letters = []
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        if letter in metrics["per_letter"]:
            if "true_positive" in metrics["per_letter"][letter]:
                available_letters.append(letter)
    
    for i, letter_actual in enumerate(available_letters):
        total_samples = metrics["per_letter"][letter_actual]["true_positive"] + metrics["per_letter"][letter_actual]["false_negative"]
        if total_samples > 0:
            detected[i, i] = metrics["per_letter"][letter_actual]["true_positive"] / total_samples
    
    plt.imshow(detected, cmap='viridis')
    plt.colorbar(label='Detection Rate')
    plt.title('Letter Detection Rate')
    plt.xticks(range(len(available_letters)), available_letters)
    plt.yticks(range(len(available_letters)), available_letters)
    plt.xlabel('Predicted Letter')
    plt.ylabel('Actual Letter')
    
    # Add text annotations
    for i in range(len(available_letters)):
        for j in range(len(available_letters)):
            if i == j:  # Only show diagonal values
                plt.text(j, i, f'{detected[i, j]:.2f}', ha='center', va='center', 
                         color='white' if detected[i, j] > 0.5 else 'black')
    
    plt.tight_layout()
    plt.savefig("test_results/letter_detection_matrix.png")
    print("Detection matrix saved to test_results/letter_detection_matrix.png")
    
if __name__ == "__main__":
    # Create test_results directory if it doesn't exist
    os.makedirs("test_results", exist_ok=True)
    
    # Visualize detection performance
    visualize_detection_performance()