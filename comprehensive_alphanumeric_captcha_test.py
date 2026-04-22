"""
Test script for the comprehensive alphanumeric captcha generator
to ensure proper inclusion of numbers in alphanumeric captchas
"""

import os
import random
import string
from PIL import Image
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import json

# Import the generator
from comprehensive_alphanumeric_captcha import ComprehensiveAlphanumericGenerator

def test_alphanumeric_digit_inclusion():
    """Test that alphanumeric captchas always include digits"""
    generator = ComprehensiveAlphanumericGenerator()
    
    # Test parameters
    modes = ['alphanumeric', 'mixed']
    case_modes = ['uppercase', 'lowercase', 'mixed']
    difficulties = ['easy', 'medium', 'hard']
    num_tests = 5  # Number of tests per configuration
    
    results = {
        'success': 0,
        'total': 0,
        'failures': [],
        'examples': []
    }
    
    # Create results directory
    os.makedirs("test_results", exist_ok=True)
    
    # Test all combinations
    for mode in modes:
        for case_mode in case_modes:
            for difficulty in difficulties:
                for i in range(num_tests):
                    # Generate captcha
                    if case_mode == 'mixed':
                        current_mode = mode
                    else:
                        # Use hyphen notation to specify case
                        current_mode = f"{mode}-{case_mode}"
                    
                    img_b64, text = generator.generate_captcha(mode=current_mode, 
                                                             length=6, 
                                                             difficulty=difficulty)
                    
                    # Check if it contains digits
                    has_digits = any(c.isdigit() for c in text)
                    digit_count = sum(1 for c in text if c.isdigit())
                    
                    # Record result
                    test_config = f"Mode: {current_mode}, Difficulty: {difficulty}, Test: {i+1}"
                    results['total'] += 1
                    
                    if has_digits and digit_count >= 2:
                        results['success'] += 1
                        print(f"✓ {test_config} - Text: {text} - Contains {digit_count} digits")
                    else:
                        results['failures'].append({
                            'config': test_config,
                            'text': text,
                            'digit_count': digit_count
                        })
                        print(f"✗ {test_config} - Text: {text} - Contains only {digit_count} digits")
                    
                    # Save a few examples
                    if len(results['examples']) < 10 and random.random() < 0.3:
                        # Convert base64 to image
                        image_data = base64.b64decode(img_b64)
                        image = Image.open(BytesIO(image_data))
                        
                        # Save example info
                        results['examples'].append({
                            'config': test_config,
                            'text': text,
                            'digit_count': digit_count,
                            'image_path': f"test_results/example_{len(results['examples'])}.png"
                        })
                        
                        # Save image
                        image.save(f"test_results/example_{len(results['examples'])-1}.png")
    
    # Print summary
    print("\n" + "=" * 40)
    print(f"Test Summary: {results['success']}/{results['total']} passed")
    print(f"Success Rate: {results['success']/results['total']*100:.1f}%")
    
    if results['failures']:
        print("\nFailures:")
        for failure in results['failures']:
            print(f"  {failure['config']} - Text: {failure['text']} - Digit count: {failure['digit_count']}")
    
    # Save results to file
    with open("test_results/digit_inclusion_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate a visual report
    if results['examples']:
        plt.figure(figsize=(15, len(results['examples']) * 3))
        
        for i, example in enumerate(results['examples']):
            try:
                img = Image.open(example['image_path'])
                plt.subplot(len(results['examples']), 1, i+1)
                plt.imshow(img)
                plt.title(f"{example['config']}\nText: {example['text']} (Digits: {example['digit_count']})")
                plt.axis('off')
            except Exception as e:
                print(f"Error displaying example {i}: {e}")
        
        plt.tight_layout()
        plt.savefig("test_results/examples_report.png")
        plt.close()
    
    return results

def test_enhanced_integration():
    """Test the enhanced integration between CAPTCHA generators"""
    try:
        from enhanced_captcha_integration import get_captcha_for_current_mode
        from flask import session
        
        print("\nTesting enhanced_captcha_integration.py...")
        print("This test requires a Flask context and may not work directly.")
        print("Use the integration in the main app to verify it's working correctly.")
    except ImportError as e:
        print(f"\nCould not test enhanced integration: {e}")
    except Exception as e:
        print(f"\nError testing enhanced integration: {e}")

if __name__ == "__main__":
    print("Testing Comprehensive Alphanumeric CAPTCHA Generator")
    print("=" * 50)
    
    # Test digit inclusion in alphanumeric captchas
    results = test_alphanumeric_digit_inclusion()
    
    # Test integration with enhanced captcha module
    test_enhanced_integration()
    
    print("\nTest completed. Check 'test_results' directory for visual examples.")