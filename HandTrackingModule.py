import cv2
import numpy as np
import time

# Global timestamp management for MediaPipe
_last_timestamp = 0

def get_monotonic_timestamp():
    """Generate strictly increasing timestamps for MediaPipe processing"""
    global _last_timestamp
    current = int(time.time() * 1000000)  # microseconds for higher precision
    if current <= _last_timestamp:
        current = _last_timestamp + 1
    _last_timestamp = current
    return current

# Try to import mediapipe, fall back to simple detection if not available
try:
    import mediapipe as mp
    # Extra check for the specific 'builder' issue
    try:
        from google.protobuf.internal import builder
        MEDIAPIPE_AVAILABLE = True
        print(f"MediaPipe loaded successfully - version: {mp.__version__}")
    except ImportError:
        print("MediaPipe found but protobuf is incompatible (missing builder)")
        MEDIAPIPE_AVAILABLE = False
        mp = None
except Exception as e:
    print(f"MediaPipe not available: {e}")
    MEDIAPIPE_AVAILABLE = False
    mp = None

class handDetector():
    def __init__(self, mode=False, maxHands=1, modelComplexity=0, detectionCon=0.7, trackCon=0.5, mediapipe_active=None):
        self.mode = mode  # Use video mode for better tracking performance
        self.maxHands = maxHands  # Limit to 1 hand for better performance
        self.modelComplex = modelComplexity  # Use lighter model for speed
        self.detectionCon = detectionCon  # Higher detection confidence for stability
        self.trackCon = trackCon
        
        # Allow manual override of MediaPipe availability
        self.mediapipe_active = MEDIAPIPE_AVAILABLE if mediapipe_active is None else mediapipe_active
        self.use_fallback = not self.mediapipe_active
        
        if self.mediapipe_active:
            self.mpHands = mp.solutions.hands
            self.hands = self.mpHands.Hands(
                static_image_mode=self.mode,  # Use video mode for smooth tracking
                max_num_hands=self.maxHands,
                model_complexity=self.modelComplex,
                min_detection_confidence=self.detectionCon,
                min_tracking_confidence=self.trackCon
            )
            self.mpDraw = mp.solutions.drawing_utils
        else:
            # Setup for fallback mode using color thresholding
            self.skin_lower = np.array([0, 48, 80], dtype=np.uint8)
            self.skin_upper = np.array([20, 255, 255], dtype=np.uint8)
            print("Using fallback hand detection mode")
        
        self.tipIds = [4, 8, 12, 16, 20]
        self.results = None
        self.lmList = []
        self.hand_center = None  # Track hand center for fallback mode
        self.finger_tip = None   # Track finger tip for fallback mode

    def findHands(self, img, draw=False):  # Disable drawing by default for performance
        if self.mediapipe_active:
            try:
                imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Use persistent hands instance for better performance and tracking
                self.results = self.hands.process(imgRGB)
                
                if self.results and self.results.multi_hand_landmarks:
                    for handLms in self.results.multi_hand_landmarks:
                        if draw:
                            # Use thinner lines for less visual clutter
                            self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS,
                                                     landmark_drawing_spec=self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2),
                                                     connection_drawing_spec=self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=1))
            except Exception as e:
                print(f"MediaPipe processing error: {e}")
                self.results = None
                self.use_fallback = True
        
        # Fallback mode using color-based hand detection
        if self.use_fallback:
            try:
                # Convert to HSV color space
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                
                # Create skin mask
                mask = cv2.inRange(hsv, self.skin_lower, self.skin_upper)
                
                # Apply some morphological operations to improve the mask
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask = cv2.erode(mask, kernel, iterations=1)
                mask = cv2.dilate(mask, kernel, iterations=2)
                mask = cv2.GaussianBlur(mask, (5, 5), 0)
                
                # Find contours in the mask
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Find the largest contour (assuming it's the hand)
                    max_contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(max_contour)
                    
                    if area > 3000:  # Minimum area to consider as a hand
                        # Get the bounding box
                        x, y, w, h = cv2.boundingRect(max_contour)
                        
                        # Calculate the center of the hand
                        self.hand_center = (x + w // 2, y + h // 2)
                        
                        # Find the highest point (assuming it's the finger tip)
                        min_y_pt = None
                        min_y = img.shape[0]  # Initialize with max possible value
                        
                        for point in max_contour[:, 0, :]:
                            if point[1] < min_y and y < point[1] < y + h and x < point[0] < x + w:
                                min_y = point[1]
                                min_y_pt = point
                        
                        if min_y_pt is not None:
                            self.finger_tip = (min_y_pt[0], min_y_pt[1])
                            
                            # Draw the hand contour, center and finger tip if requested
                            if draw:
                                cv2.drawContours(img, [max_contour], -1, (0, 255, 0), 2)
                                cv2.circle(img, self.hand_center, 5, (0, 0, 255), -1)
                                cv2.circle(img, self.finger_tip, 8, (255, 0, 0), -1)
                                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    else:
                        self.hand_center = None
                        self.finger_tip = None
                else:
                    self.hand_center = None
                    self.finger_tip = None
            
            except Exception as e:
                print(f"Fallback hand detection error: {e}")
                self.hand_center = None
                self.finger_tip = None
                
            # Show fallback mode indicator
            cv2.putText(img, "Fallback Mode Active", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        try:
            if self.mediapipe_active and self.results and self.results.multi_hand_landmarks:
                if handNo < len(self.results.multi_hand_landmarks):
                    myHand = self.results.multi_hand_landmarks[handNo]
                    for id, lm in enumerate(myHand.landmark):
                        h, w, c = img.shape
                        cx, cy = int(lm.x*w), int(lm.y*h)
                        self.lmList.append([id, cx, cy])
                        if draw:
                            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
            elif self.use_fallback and self.finger_tip is not None and self.hand_center is not None:
                # In fallback mode, we only care about index finger tip (id 8) for drawing
                # and the base of the hand for reference
                self.lmList = [
                    [0, self.hand_center[0], self.hand_center[1]],  # Hand center as point 0
                    [8, self.finger_tip[0], self.finger_tip[1]]     # Finger tip as point 8 (index finger)
                ]
                
                # Add some additional reference points based on the hand center and finger tip
                # These help the fingersUp function work somewhat with the fallback mode
                base_x, base_y = self.hand_center
                tip_x, tip_y = self.finger_tip
                
                # Add a point for thumb (id 4)
                self.lmList.append([4, base_x - 30, base_y])
                
                # Add more finger reference points
                self.lmList.append([5, base_x, base_y - 20])  # Base of index finger
                self.lmList.append([6, (base_x + tip_x) // 2, (base_y + tip_y) // 2])  # Middle of index finger
                
                if draw:
                    for point in self.lmList:
                        cv2.circle(img, (point[1], point[2]), 5, (255, 0, 0), cv2.FILLED)
        except Exception as e:
            print(f"Position detection error: {e}")
            self.lmList = []
        return self.lmList

    def fingersUp(self):
        fingers = []
        
        if self.mediapipe_active and len(self.lmList) >= 21:  # Full MediaPipe hand has 21 landmarks
            # Standard MediaPipe-based detection
            # Thumb
            if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0]-1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            # Four fingers
            for id in range(1, 5):
                if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id]-2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        elif self.use_fallback and len(self.lmList) > 0:
            # In fallback mode with detected hand
            if self.finger_tip is not None and self.hand_center is not None:
                # Check if finger tip is significantly above the hand center (index finger up)
                if self.finger_tip[1] < self.hand_center[1] - 30:  # Y decreases as you go up
                    fingers = [0, 1, 0, 0, 0]  # Assume only index finger is up for drawing
                else:
                    fingers = [0, 0, 0, 0, 0]  # No fingers up
            else:
                fingers = [0, 0, 0, 0, 0]  # Default to no fingers up
        else:
            # Default finger state if no detection
            fingers = [0, 1, 0, 0, 0]  # Only index finger up for compatibility
            
        return fingers
#         cv2.waitKey(1)
#
# def __name__ == "__main__":
#     main()
