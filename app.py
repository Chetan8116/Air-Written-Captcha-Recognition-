from flask import Flask, redirect, url_for, render_template, Response, request, session, flash, jsonify
import cv2
import numpy as np
import random
from captcha_utils import generate_captcha_image, generate_numeric_captcha
# Import enhanced captcha integration
from enhanced_captcha_integration import (
    generate_enhanced_numeric_captcha, 
    get_captcha_for_current_mode, 
    load_realistic_captcha_model, 
    get_comprehensive_generator
)

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Needed for session

# Import functions from VirtualPainter
from VirtualPainter import generate_frames, save_drawing, cleanup_camera, set_recognition_mode, get_current_recognition, clear_canvas, get_current_label, get_current_drawing_mode, set_drawing_mode

# Register the video feed route
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/save_drawing', methods=['POST'])
def save_drawing_route():
    return save_drawing(request)

@app.route('/set_mode', methods=['POST'])
def set_mode():
    """Set numeric-only recognition mode"""
    mode = request.form.get('mode') or (request.json and request.json.get('mode')) or request.args.get('mode') or 'off'

    if mode not in ['num', 'off']:
        if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return {"success": False, "error": "Only numeric mode is available"}, 400
        flash("Only numeric mode is available", "warning")
        return redirect(url_for('home'))

    set_recognition_mode(mode)
    session["numeric_captcha"] = generate_numeric_captcha(6)
    session["captcha_type"] = "numeric"

    if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return {
            "success": True,
            "mode": mode,
            "current_predict_mode": mode,
            "alphabet_mode": "auto"
        }

    return redirect(url_for('home'))

@app.route('/clear_canvas', methods=['GET', 'POST'])
def clear_canvas_route():
    """Clear the drawing canvas"""
    clear_canvas()
    
    # Check if it's an AJAX request
    if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({"success": True, "message": "Canvas cleared"})
    
    # Traditional form submission fallback
    flash("Canvas cleared", "info")
    return redirect(url_for('home'))

@app.route('/verify_captcha', methods=['POST'])
def verify_captcha():
    """Verify the captcha with recognized characters"""
    recognized_input = request.form.get('recognized_input', '').strip()
    expected_captcha = session.get('numeric_captcha', '')

    if recognized_input == expected_captcha:
        flash("Numeric captcha verification successful! ✓", "success")
        session["numeric_captcha"] = generate_numeric_captcha(6)
    else:
        flash(f"Captcha verification failed. Expected: {expected_captcha}, Got: {recognized_input}", "danger")
    
    return redirect(url_for('home'))

@app.route('/regenerate_captcha', methods=['POST'])
def regenerate_captcha():
    """Generate a new captcha with random style based on current mode"""
    captcha_type = session.get('captcha_type', 'numeric')
    print(f"Regenerating captcha: captcha_type={captcha_type}")
    
    # Use the enhanced captcha integration
    captcha_img, new_captcha = get_captcha_for_current_mode(captcha_type)
    
    if request.is_json:
        return jsonify({
            'success': True,
            'captcha_img': captcha_img
        })
    else:
        flash("New captcha generated", "info")
        return redirect(url_for('home'))

@app.route('/refresh_camera', methods=['POST'])
def refresh_camera():
    """Refresh camera connection"""
    try:
        from camera_reset import reset_camera
        success = reset_camera()
        if success:
            flash("Camera refreshed successfully!", "success")
        else:
            flash("Camera refresh attempted, please try again if issues persist", "warning")
    except Exception as e:
        flash(f"Camera refresh failed: {e}", "error")
    return redirect(url_for('home'))

@app.route("/")
def home():
    """Home page - Air Writing Recognition feature"""
    from VirtualPainter import get_current_recognition
    current_recognition = get_current_recognition()
    current_mode = current_recognition.get("mode", "off")
    actual_mode = current_recognition.get("actual_mode", current_mode)
    alphabet_mode = "auto"

    if current_mode not in ['num', 'off']:
        current_mode = 'off'
    if actual_mode not in ['num', 'off']:
        actual_mode = 'off'

    print(f"HOME: mode={current_mode}, actual_mode={actual_mode}")
    
    current_char = get_current_label()
    # Ensure mode is properly initialized to 'off' if not set
    if current_mode is None or current_mode == "":
        current_mode = "off"
    
    captcha_text = generate_numeric_captcha(6)
    session["numeric_captcha"] = captcha_text
    session["captcha_type"] = "numeric"
    
    # Focus on readable styles for better user experience
    captcha_styles = ['modern', 'gradient', 'shadow', 'neon', 'retro', 'sketch', 'fire', 'ice']
    captcha_style = random.choice(captcha_styles)
    session["captcha_style"] = captcha_style
    captcha_img = generate_captcha_image(captcha_text, captcha_style)
    
    return render_template("feature.html", 
                         recognition_mode=current_mode,
                         recognized_char=current_char,
                         captcha_img=captcha_img,
                         alphabet_mode=alphabet_mode,
                         actual_mode=actual_mode)

# CAPTCHA routes
@app.route("/captcha", methods=["GET", "POST"])
def captcha():
    if request.method == "POST":
        user_input = request.form.get("captcha_input", "")
        expected_captcha = session.get("numeric_captcha", "")

        if user_input == expected_captcha:
            flash("CAPTCHA verified successfully!", "success")
            return redirect(url_for("home"))
        else:
            flash(f"Incorrect CAPTCHA. Please try again.", "danger")
            return redirect(url_for("captcha"))
            
    # Generate a new captcha using enhanced captcha integration
    captcha_img, captcha_text = get_captcha_for_current_mode()
    return render_template("captcha.html", captcha_img=captcha_img)


@app.route("/feature")
def feature():
    """Legacy route - redirects to home"""
    return redirect(url_for("home"))


@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route('/get_recognized_char')
def get_recognized_char():
    """Get current recognized character for dynamic updates"""
    current_char = get_current_label()
    current_mode = get_current_recognition()
    return jsonify({
        'recognized_char': current_char or 'None',
        'recognition_mode': current_mode or 'off'
    })

@app.route('/toggle_realistic_captcha', methods=['POST'])
def toggle_realistic_captcha():
    """Toggle between realistic and enhanced numeric CAPTCHAs"""
    # Toggle the current setting
    current_setting = session.get('use_realistic_captcha', True)  # Default to realistic
    session['use_realistic_captcha'] = not current_setting
    
    # Generate a new captcha with the updated setting
    captcha_img, captcha_text = get_captcha_for_current_mode()
    
    # Return the response based on request type
    if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({
            'success': True,
            'use_realistic_captcha': not current_setting,
            'captcha_img': captcha_img
        })
    else:
        mode_text = "Realistic" if not current_setting else "Enhanced"
        flash(f"Switched to {mode_text} CAPTCHA mode", "info")
        return redirect(url_for('captcha'))

@app.route('/toggle_comprehensive_captcha', methods=['POST'])
def toggle_comprehensive_captcha():
    """Toggle between comprehensive and traditional CAPTCHA generators"""
    # Toggle the current setting
    current_setting = session.get('use_comprehensive_captcha', True)  # Default to comprehensive
    session['use_comprehensive_captcha'] = not current_setting
    
    # Get or initialize the comprehensive generator
    if current_setting == False and get_comprehensive_generator() is None:
        try:
            # First time enabling - initialize the generator
            from enhanced_captcha_integration import get_comprehensive_generator
            if get_comprehensive_generator() is None:
                flash("Unable to initialize comprehensive CAPTCHA generator", "warning")
                return redirect(url_for('captcha'))
        except Exception as e:
            flash(f"Error initializing comprehensive CAPTCHA generator: {e}", "danger")
            return redirect(url_for('captcha'))
    
    # Generate a new captcha with the updated setting
    captcha_img, captcha_text = get_captcha_for_current_mode()
    
    # Return the response based on request type
    if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({
            'success': True,
            'use_comprehensive_captcha': not current_setting,
            'captcha_img': captcha_img
        })
    else:
        mode_text = "Comprehensive" if not current_setting else "Traditional"
        flash(f"Switched to {mode_text} CAPTCHA mode", "info")
        return redirect(url_for('captcha'))

if __name__ == "__main__":
    print("Loading models and initializing camera...")
    
    # Preload the realistic captcha model
    realistic_model = load_realistic_captcha_model()
    if realistic_model:
        print("Realistic CAPTCHA model loaded successfully!")
    else:
        print("Warning: Realistic CAPTCHA model could not be loaded, will use enhanced model instead.")
    
    # Preload the comprehensive generator
    comprehensive_generator = get_comprehensive_generator()
    if comprehensive_generator:
        print("Comprehensive numeric CAPTCHA generator loaded successfully!")
    else:
        print("Warning: Comprehensive CAPTCHA generator could not be loaded, will use traditional generators instead.")
    
    print("Models loaded successfully!")
    print("Starting Flask app...")
    app.run(debug=True)
