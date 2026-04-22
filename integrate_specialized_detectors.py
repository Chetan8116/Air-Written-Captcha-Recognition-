"""
Integration script for specialized letter detectors in VirtualPainter

This script adds code to the VirtualPainter application to use
specialized letter detectors for all uppercase letters A-Z.
"""

import os
import sys
import re

def integrate_specialized_detectors():
    """Add specialized letter detector code to VirtualPainter.py"""
    virtual_painter_path = "VirtualPainter.py"
    
    if not os.path.exists(virtual_painter_path):
        print(f"Error: {virtual_painter_path} not found")
        return False
    
    # Read the current file
    with open(virtual_painter_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Create backup
    backup_path = "VirtualPainter.py.specialized_backup"
    with open(backup_path, 'w', encoding='utf-8') as file:
        file.write(content)
    
    print(f"Created backup at {backup_path}")
    
    # Check if the integration has already been done
    if "from integrated_letter_detection import" in content:
        print("Specialized letter detectors already integrated")
        return True
    
    # Add import statement
    import_pattern = "from specialized_letter_detection import detect_specific_letter"
    if import_pattern in content:
        # Replace with the integrated version
        new_import = "from integrated_letter_detection import analyze_and_detect_letter, load_letter_models"
        content = content.replace(import_pattern, new_import)
    else:
        # Add after other imports
        import_section = "# Import the B detector functionality\n"
        new_import = import_section + "from integrated_letter_detection import analyze_and_detect_letter, load_letter_models\n"
        content = content.replace(import_section, new_import)
    
    # Add loading code for letter models
    init_pattern = "# Initialize B detector model\nload_b_detector_model()"
    if init_pattern in content:
        # Replace with integrated version
        new_init = "# Initialize letter detector models\nload_letter_models()"
        content = content.replace(init_pattern, new_init)
    else:
        # Look for alternate initialization patterns
        alternate_pattern = "# Load specialized models\ntry:"
        new_init = "# Load specialized letter detector models\nload_letter_models()\n\n# Load specialized models\ntry:"
        if alternate_pattern in content:
            content = content.replace(alternate_pattern, new_init)
    
    # Update the B detection logic to use integrated detection
    b_detection_pattern = "                                # Special case for B recognition using our trained B detector model"
    new_b_detection = """                                # Use comprehensive letter detection with specialized models
                                image_for_detection = large_img.copy()
                                detected_letter, detection_confidence = analyze_and_detect_letter(image_for_detection, uppercase_only=True)
                                
                                if detected_letter:
                                    print(f"Specialized detection found letter {detected_letter} with confidence {detection_confidence:.4f}")
                                    final_char = detected_letter
                                    final_confidence = max(confidence, detection_confidence)
                                    
                                    # Additional checks for specific letters
                                    if detected_letter == 'B':
                                        # Special case for B
                                        print(f"Using specialized B detection with confidence {detection_confidence:.4f}")
                                    elif detected_letter in ['D', 'P', 'R']:
                                        # These letters are often confused
                                        print(f"Detected commonly confused letter {detected_letter}")
                                """
    if b_detection_pattern in content:
        # Find the whole B detection block
        b_detection_block_pattern = b_detection_pattern + r"[\s\S]+?(?=\s{32}\w)"
        b_detection_match = re.search(b_detection_block_pattern, content)
        if b_detection_match:
            content = content.replace(b_detection_match.group(0), new_b_detection)
    
    # Update confused letter detection
    confused_pattern = "                            # Check for specific letters that are confused with others"
    if confused_pattern in content:
        new_confused = """                            # Use integrated letter detection for commonly confused letters
                            image_for_detection = large_img.copy()
                            detected_letter, detection_confidence = analyze_and_detect_letter(image_for_detection, uppercase_only=True)
                            
                            if detected_letter and detection_confidence > 0.6:
                                # Override with high confidence detection
                                final_char = detected_letter
                                final_confidence = detection_confidence
                                print(f"Using integrated letter detection: {final_char} with confidence {final_confidence:.4f}")
                            """
        content = content.replace(confused_pattern, new_confused)
    
    # Write the updated content
    with open(virtual_painter_path, 'w', encoding='utf-8') as file:
        file.write(content)
    
    print(f"Successfully integrated specialized letter detectors into {virtual_painter_path}")
    return True

if __name__ == "__main__":
    if integrate_specialized_detectors():
        print("Integration successful!")
    else:
        print("Integration failed!")