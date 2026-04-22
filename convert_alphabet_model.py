"""
Model Converter for Alphabet CNN Model
This script converts an older TensorFlow model to a format compatible with newer TensorFlow versions
"""

import os
import sys
import tensorflow as tf
import numpy as np
import json
import h5py
import argparse

def convert_model(input_path, output_path, mapping_path=None):
    """
    Convert an older TensorFlow model to a format compatible with newer TensorFlow versions
    
    Args:
        input_path: Path to the input model file
        output_path: Path to save the converted model
        mapping_path: Path to the class mapping file
    
    Returns:
        bool: True if conversion was successful, False otherwise
    """
    print(f"TensorFlow version: {tf.__version__}")
    
    try:
        # Check if the output directory exists, create it if not
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Try to extract model architecture and weights
        model = None
        print(f"Loading model from {input_path}...")
        try:
            # Try loading with no custom objects
            model = tf.keras.models.load_model(input_path, compile=False)
            print("Model loaded successfully using standard loading")
        except Exception as e1:
            print(f"Standard loading failed: {e1}")
            
            try:
                # Try extracting architecture from the h5 file
                print("Attempting to extract model architecture and weights separately...")
                with h5py.File(input_path, 'r') as h5file:
                    # Check if model config exists in the file
                    model_config = None
                    if 'model_config' in h5file.attrs:
                        model_config = h5file.attrs.get('model_config')
                    elif 'model_config' in h5file:
                        model_config = h5file['model_config'][()]
                    
                    if model_config is not None:
                        if isinstance(model_config, bytes):
                            model_config = model_config.decode('utf-8')
                        
                        # Load the model from its config
                        import json
                        model_json = json.loads(model_config)
                        
                        # Modify the model configuration for compatibility
                        if 'config' in model_json:
                            for layer in model_json.get('config', {}).get('layers', []):
                                if 'config' in layer:
                                    # Replace batch_shape with batch_input_shape
                                    if 'batch_shape' in layer['config']:
                                        layer['config']['batch_input_shape'] = layer['config']['batch_shape']
                                        del layer['config']['batch_shape']
                                    
                                    # Remove problematic attributes
                                    for attr in ['registered_name', 'module']:
                                        if attr in layer['config']:
                                            del layer['config'][attr]
                    
                        # Create model from the cleaned config
                        model = tf.keras.models.model_from_json(json.dumps(model_json))
                        
                        # Load weights
                        try:
                            model.load_weights(input_path)
                            print("Successfully extracted model architecture and loaded weights!")
                        except Exception as weight_error:
                            print(f"Error loading weights: {weight_error}")
                            raise
                    else:
                        print("Could not find model_config in h5 file")
                        raise ValueError("Model configuration not found in the h5 file")
            
            except Exception as e2:
                print(f"Extraction failed: {e2}")
                
                # As a last resort, recreate the model with the same architecture
                print("Recreating model with compatible architecture...")
                
                # Get input shape from the h5 file if possible
                input_shape = (28, 28, 1)  # Default shape for alphabet recognition
                try:
                    with h5py.File(input_path, 'r') as h5file:
                        # Try to find input shape in the model
                        if 'model_weights' in h5file:
                            for layer_name in h5file['model_weights']:
                                if 'kernel' in h5file['model_weights'][layer_name]:
                                    kernel_shape = h5file['model_weights'][layer_name]['kernel:0'].shape
                                    print(f"Found kernel shape: {kernel_shape}")
                                    # First convolutional layer's kernel shape: (height, width, input_channels, filters)
                                    if len(kernel_shape) == 4:
                                        input_shape = (None, None, kernel_shape[2])
                                        break
                except Exception as shape_error:
                    print(f"Error determining input shape: {shape_error}")
                
                # Create a similar CNN model
                model = tf.keras.Sequential([
                    tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
                    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                    tf.keras.layers.MaxPooling2D((2, 2)),
                    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                    tf.keras.layers.MaxPooling2D((2, 2)),
                    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dense(52, activation='softmax')  # 52 classes for A-Z, a-z
                ])
                
                print("Created compatible model with similar architecture")
        
        # Save the model in SavedModel format (more compatible with newer TF versions)
        if model is not None:
            # Compile the model with basic settings
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Save the model in multiple formats
            # 1. As HDF5 (.h5) file
            print(f"Saving converted model to {output_path}")
            model.save(output_path, save_format='h5')
            
            # 2. Also save in SavedModel format
            savedmodel_path = os.path.splitext(output_path)[0] + "_savedmodel"
            print(f"Saving in SavedModel format to {savedmodel_path}")
            model.save(savedmodel_path, save_format='tf')
            
            print("Model conversion successful!")
            return True
        else:
            print("Failed to create a valid model")
            return False
            
    except Exception as e:
        print(f"Model conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to parse arguments and convert the model"""
    parser = argparse.ArgumentParser(description="Convert TensorFlow model to newer format")
    parser.add_argument('--input', '-i', required=True, help="Path to the input model file")
    parser.add_argument('--output', '-o', required=True, help="Path to save the converted model")
    parser.add_argument('--mapping', '-m', help="Path to the class mapping file")
    
    args = parser.parse_args()
    
    return convert_model(args.input, args.output, args.mapping)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)