import os
import sys
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Import specialized classifier utility functions
from specialized_classifier_utils import enhance_air_writing_features, predict_with_specialized_model

def integrate_specialized_classifier():
    """
    Integrate the specialized 5 vs 3 classifier into the VirtualPainter module.
    
    This function:
    1. Checks if the specialized classifier exists
    2. Creates a backup of the VirtualPainter.py file
    3. Modifies the VirtualPainter.py file to use the specialized classifier
    """
    # Check if specialized classifier exists
    if not os.path.exists("5_vs_3_model.h5"):
        print("Specialized classifier not found.")
        print("Please run train_5_vs_3_classifier.py first.")
        return False
    
    # Create backup of VirtualPainter.py
    if not os.path.exists("VirtualPainter.py.bak"):
        print("Creating backup of VirtualPainter.py...")
        with open("VirtualPainter.py", "r") as src:
            with open("VirtualPainter.py.bak", "w") as dst:
                dst.write(src.read())
    
    # Load and test the specialized classifier
    try:
        specialized_model = load_model("5_vs_3_model.h5")
        print("Successfully loaded specialized classifier.")
    except Exception as e:
        print(f"Error loading specialized classifier: {e}")
        return False
    
    # Read VirtualPainter.py
    with open("VirtualPainter.py", "r") as f:
        lines = f.readlines()
    
    # Find the global variable section to add the specialized classifier imports
    # Look for a spot after imports but before function definitions
    import_section_end = 0
    first_function_def = len(lines)
    
    for i, line in enumerate(lines):
        if "import" in line and i > import_section_end:
            import_section_end = i
        if line.startswith("def ") and i < first_function_def:
            first_function_def = i
    
    # Insert the specialized classifier imports at a safe location
    insert_position = min(import_section_end + 5, first_function_def)
    
    # Ensure we insert in a safe position (after imports, before functions)
    while insert_position < first_function_def:
        if not lines[insert_position].strip() or lines[insert_position].startswith("#"):
            break
        insert_position += 1
    
    # Insert the specialized classifier imports
    insert_imports = [
        "\n# Import specialized classifier for 5 vs 3 distinction\n",
        "SPECIALIZED_MODEL = None\n",
        "try:\n",
        "    from specialized_classifier_utils import enhance_air_writing_features, predict_with_specialized_model\n",
        "    SPECIALIZED_MODEL = tf.keras.models.load_model('5_vs_3_model.h5')\n",
        "    print(\"Specialized 5 vs 3 classifier loaded successfully\")\n",
        "except Exception as e:\n",
        "    print(f\"Error loading specialized 5 vs 3 classifier: {e}\")\n",
        "    SPECIALIZED_MODEL = None\n",
        "\n"
    ]
    
    lines = lines[:insert_position] + insert_imports + lines[insert_position:]
    
    # Find the prediction function and modify it to use the specialized classifier
    for i, line in enumerate(lines):
        if "def predict_character():" in line:
            prediction_function_start = i
            break
    
    # Find the end of the prediction function
    prediction_function_end = prediction_function_start
    bracket_count = 0
    for i in range(prediction_function_start, len(lines)):
        if lines[i].strip().startswith("def "):
            prediction_function_end = i - 1
            break
    
    # Modify the prediction function to use the specialized classifier
    modified_prediction_function = [
        "def predict_character():\n",
        "    \"\"\"Optimized character prediction function with specialized 5 vs 3 classifier\"\"\"\n",
        "    global imgCanvas, PREDICT, AlphaMODEL, NumMODEL, AlphaLABELS, NumLABELS, label, MODELS_AVAILABLE, SPECIALIZED_MODEL\n",
        "    if imgCanvas is None or PREDICT == \"off\" or not MODELS_AVAILABLE:\n",
        "        return \"\"\n",
        "    \n",
        "    try:\n",
        "        # Quick check if canvas has any content (optimization)\n",
        "        if np.sum(imgCanvas) < 1000:  # Not enough drawn content\n",
        "            return \"\"\n",
        "        \n",
        "        # Preprocess the canvas for prediction with optimizations\n",
        "        gray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)\n",
        "        \n",
        "        # Use INTER_AREA for better downsampling quality\n",
        "        resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)\n",
        "        \n",
        "        # Apply light Gaussian blur to smooth edges for better recognition\n",
        "        blurred = cv2.GaussianBlur(resized, (3, 3), 0)\n",
        "        \n",
        "        # Save the processed image for debugging\n",
        "        cv2.imwrite('last_processed_digit.png', blurred)\n",
        "        \n",
        "        # Use enhanced preprocessing for better feature extraction\n",
        "        if 'enhance_air_writing_features' in globals():\n",
        "            enhanced = enhance_air_writing_features(blurred)\n",
        "            cv2.imwrite('last_enhanced_digit.png', enhanced)\n",
        "            blurred = enhanced\n",
        "        \n",
        "        # Normalize efficiently\n",
        "        normalized = blurred.astype(np.float32) / 255.0\n",
        "        \n",
        "        # Reshape for model input: (batch, height, width, channels)\n",
        "        input_img = normalized.reshape(1, 28, 28, 1)\n",
        "        \n",
        "        # Predict using the appropriate model based on current mode\n",
        "        predicted_class = None\n",
        "        confidence = 0.0\n",
        "        \n",
        "        if PREDICT == \"alpha\" and AlphaMODEL is not None:\n",
        "            predictions = AlphaMODEL.predict(input_img, verbose=0)\n",
        "            predicted_class = int(np.argmax(predictions, axis=1)[0])\n",
        "            confidence = float(np.max(predictions))\n",
        "            return AlphaLABELS.get(predicted_class, \"\")\n",
        "        elif PREDICT == \"num\" and NumMODEL is not None:\n",
        "            predictions = NumMODEL.predict(input_img, verbose=0)\n",
        "            predicted_class = int(np.argmax(predictions, axis=1)[0])\n",
        "            confidence = float(np.max(predictions))\n",
        "            \n",
        "            # Use specialized classifier for 3 vs 5 distinction\n",
        "            if predicted_class in [3, 5] and SPECIALIZED_MODEL is not None:\n",
        "                # Get prediction from specialized model\n",
        "                specialized_pred = SPECIALIZED_MODEL.predict(input_img, verbose=0)[0][0]\n",
        "                # If it's a 5 with high confidence or a 3 with low confidence from specialized model\n",
        "                if specialized_pred > 0.7:  # Higher threshold for '5'\n",
        "                    predicted_class = 5\n",
        "                elif specialized_pred < 0.3:  # Lower threshold for '3'\n",
        "                    predicted_class = 3\n",
        "                # Otherwise keep the original prediction but with higher confidence\n",
        "                \n",
        "            return NumLABELS.get(predicted_class, \"\")\n",
        "        elif PREDICT == \"alphanum\":\n",
        "            # Use the new trained alphanumeric model if available, otherwise fallback to dual model approach\n",
        "            if AlphanumericMODEL is not None:\n",
        "                try:\n",
        "                    # Convert the processed image back to uint8 format for the alphanumeric model\n",
        "                    alphanum_image = (input_img.reshape(28, 28) * 255).astype(np.uint8)\n",
        "                    char, confidence = predict_alphanumeric_character(alphanum_image)\n",
        "                    if char and confidence > 0.1:  # Minimum confidence threshold\n",
        "                        # Check if the prediction is a digit and might be confused between 3 and 5\n",
        "                        if char in ['3', '5'] and SPECIALIZED_MODEL is not None:\n",
        "                            specialized_pred = SPECIALIZED_MODEL.predict(input_img, verbose=0)[0][0]\n",
        "                            if specialized_pred > 0.7:\n",
        "                                return '5'\n",
        "                            elif specialized_pred < 0.3:\n",
        "                                return '3'\n",
        "                        return str(char)\n",
        "                    else:\n",
        "                        return \"\"\n",
        "                except Exception as e:\n",
        "                    print(f\"Alphanumeric recognition error: {e}\")\n",
        "                    return \"\"\n",
        "            elif AlphaMODEL is not None and NumMODEL is not None:\n",
        "                # Fallback: Try both models and select the one with higher confidence\n",
        "                alpha_predictions = AlphaMODEL.predict(input_img, verbose=0)\n",
        "                num_predictions = NumMODEL.predict(input_img, verbose=0)\n",
        "                \n",
        "                alpha_confidence = np.max(alpha_predictions)\n",
        "                num_confidence = np.max(num_predictions)\n",
        "                \n",
        "                if alpha_confidence > num_confidence:\n",
        "                    predicted_class = int(np.argmax(alpha_predictions, axis=1)[0])\n",
        "                    return AlphaLABELS.get(predicted_class, \"\")\n",
        "                else:\n",
        "                    predicted_class = int(np.argmax(num_predictions, axis=1)[0])\n",
        "                    # Use specialized classifier for 3 vs 5 distinction\n",
        "                    if predicted_class in [3, 5] and SPECIALIZED_MODEL is not None:\n",
        "                        specialized_pred = SPECIALIZED_MODEL.predict(input_img, verbose=0)[0][0]\n",
        "                        if specialized_pred > 0.7:\n",
        "                            return '5'\n",
        "                        elif specialized_pred < 0.3:\n",
        "                            return '3'\n",
        "                    return NumLABELS.get(predicted_class, \"\")\n",
        "            else:\n",
        "                return \"\"\n",
        "        else:\n",
        "            return \"\"\n",
        "    except Exception as e:\n",
        "        print(f\"Prediction error: {e}\")\n",
        "        return \"\"\n",
        "\n"
    ]
    
    # Replace the prediction function
    lines = lines[:prediction_function_start] + modified_prediction_function + lines[prediction_function_end+1:]
    
    # Write the modified file
    with open("VirtualPainter.py", "w") as f:
        f.writelines(lines)
    
    print("Successfully integrated specialized classifier into VirtualPainter.py")
    print("You can now run the application with improved 5 vs 3 recognition.")
    return True

if __name__ == "__main__":
    print("Integrating specialized classifier for better 5 vs 3 distinction...")
    integrate_specialized_classifier()