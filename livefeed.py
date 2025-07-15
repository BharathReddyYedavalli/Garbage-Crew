from model import get_transform, classes
import cv2
import torch
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import numpy as np

# Load model
model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load("waste_classifier_resnet18.pth", map_location=torch.device('cpu')))
# SET CPU TO 'CUDA' IF YOU WANT TO USE GPU
# CUDA IS ONLY COMPATIBLE WITH PYTHON 3.11 AND BELOW, HASN'T UPDATED TO 3.12 OR 3.13 YET

model.eval()
MIN_AREA_THRESHOLD = 15000  # min area for detection
MAX_AREA_THRESHOLD = 35000  # max area for detection
CONFIDENCE_THRESHOLD = .7  # min confidence for classification
TRACKING_ENABLED = True  # set to true automatically, but can be toggled with 't' key
DEBUG_MODE = False  # debug mode, toggle with 'd' key to see different masks

# checks to ignore static background
backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

# grab transform function from model.py
transform = get_transform()

# begin video capture
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

        # get frame dimensions from video capture to later resize
    height, width = frame.shape[:2]
    frame_area = height * width
    
    # copying frame for processing
    processed_frame = frame.copy()
    
    if TRACKING_ENABLED:
        # motion detection for moving objects
        fgMask_motion = backSub.apply(frame)
        
        # static object detection using edge detection and thresholding
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # adaptive thresholding for different lighting conditions
        adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY_INV, 11, 2)
        
        # edge detection
        edges = cv2.Canny(blurred, 30, 100)
        
        # combine edge detection with adaptive threshold
        static_mask = cv2.bitwise_or(adaptive_thresh, edges)
        
        # clean up both masks
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # clean motion mask
        fgMask_motion = cv2.morphologyEx(fgMask_motion, cv2.MORPH_OPEN, kernel)
        fgMask_motion = cv2.morphologyEx(fgMask_motion, cv2.MORPH_CLOSE, kernel)
        
        # clean static mask
        static_mask = cv2.morphologyEx(static_mask, cv2.MORPH_CLOSE, kernel)
        static_mask = cv2.morphologyEx(static_mask, cv2.MORPH_OPEN, kernel)
        
        # combine both masks to detect both moving and stationary objects
        combined_mask = cv2.bitwise_or(fgMask_motion, static_mask)
        
        # apply additional morphological operations to connect nearby regions
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.dilate(combined_mask, kernel_large, iterations=1)
        combined_mask = cv2.erode(combined_mask, kernel_large, iterations=1)
        
        # find contours from the combined mask
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # debug: show detection masks
        if DEBUG_MODE:
            cv2.imshow("Motion Detection", fgMask_motion)
            cv2.imshow("Static Detection", static_mask)
            cv2.imshow("Combined Detection", combined_mask)
        
        # process each contour
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # check if area meets threshold requirements
            if MIN_AREA_THRESHOLD <= area <= MAX_AREA_THRESHOLD:
                # get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # calculate area percentage of the frame
                area_percentage = (area / frame_area) * 100
                
                # extract region of interest for classification
                roi = frame[y:y+h, x:x+w]
                if roi.size > 0:
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    input_tensor = transform(roi_rgb).unsqueeze(0)
                    
                    with torch.no_grad():
                        output = model(input_tensor)
                        probabilities = torch.nn.functional.softmax(output, dim=1)
                        confidence, predicted = torch.max(probabilities, 1)
                        
                        # only show classification if confidence is above threshold
                        if confidence.item() > CONFIDENCE_THRESHOLD:
                            label = classes[predicted.item()]
                            confidence_pct = confidence.item() * 100

                            # CATEGORY-SPECIFIC ACTIONS
                            if label == "Recycle":
                                print("Recycle detected!")
                                motor_recycle.on_for_degrees(SpeedPercent(50), 90)  # Example: turn 90 degrees
                                sound.speak("Recycle")
                            elif label == "Compost":
                                print("Compost detected!")
                                motor_compost.on_for_degrees(SpeedPercent(50), 90)
                                sound.speak("Compost")
                            elif label == "Other":
                                print("Other detected!")
                                # You can assign a motor or just play a sound
                                sound.speak("Other")
                            elif label == "Trashes":
                                print("Trash detected!")
                                motor_trash.on_for_degrees(SpeedPercent(50), 90)
                                sound.speak("Trash")
                            
                            # draw bounding rectangle
                            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            
                            # prepare label text
                            label_text = f'{label}: {confidence_pct:.1f}%'
                            area_text = f'Area: {area_percentage:.1f}%'
                            
                            # draw label background
                            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(processed_frame, (x, y - text_height - 30), (x + max(text_width, len(area_text) * 10), y), (0, 255, 0), -1)
                            
                            # draw text
                            cv2.putText(processed_frame, label_text, (x, y - text_height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                            cv2.putText(processed_frame, area_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                        else:
                            # draw rectangle for detected object but low confidence "red"
                            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                            cv2.putText(processed_frame, f'Low conf: {confidence.item()*100:.1f}%', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    else:
        # fallback: classify entire frame (original behavior)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = transform(frame_rgb).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            label = classes[predicted.item()]
            confidence_pct = confidence.item() * 100

        cv2.putText(processed_frame, f'{label}: {confidence_pct:.1f}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    
    # show threshold info
    cv2.putText(processed_frame, f'Min Area: {MIN_AREA_THRESHOLD} | Max Area: {MAX_AREA_THRESHOLD}', (10, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(processed_frame, f'Click + to increase min area, - to decrease', (10, height - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(processed_frame, f'Click ] to increase max area, [ to decrease', (10, height - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(processed_frame, f'Confidence Threshold: {CONFIDENCE_THRESHOLD:.1f} | Tracking: {"ON" if TRACKING_ENABLED else "OFF"}', (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(processed_frame, 'Press T to toggle tracking, D for debug, Q to quit', (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    if DEBUG_MODE:
        # different masks for debugging
        cv2.imshow("Foreground Mask - Motion Detection", fgMask_motion)
        cv2.imshow("Static Mask - Edge Detection", static_mask)
        cv2.imshow("Combined Mask", combined_mask)
    
    cv2.imshow("Waste Classification with Tracking", processed_frame)

    # key presses for interaction
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('t'):
        TRACKING_ENABLED = not TRACKING_ENABLED
        print(f"Tracking {'enabled' if TRACKING_ENABLED else 'disabled'}")
    elif key == ord('d'):
        DEBUG_MODE = not DEBUG_MODE
        print(f"Debug mode {'enabled' if DEBUG_MODE else 'disabled'}")
    elif key == ord('+') or key == ord('='):
        MIN_AREA_THRESHOLD += 250
        print(f"Min area threshold: {MIN_AREA_THRESHOLD}")
    elif key == ord('-'):
        MIN_AREA_THRESHOLD = max(500, MIN_AREA_THRESHOLD - 250)
        print(f"Min area threshold: {MIN_AREA_THRESHOLD}")
    elif key == ord(']'):
        MAX_AREA_THRESHOLD += 250
        print(f"Max area threshold: {MAX_AREA_THRESHOLD}")
    elif key == ord('['):
        MAX_AREA_THRESHOLD = max(1000, MAX_AREA_THRESHOLD - 250)
        print(f"Max area threshold: {MAX_AREA_THRESHOLD}")
    elif key == ord('d'):
        DEBUG_MODE = not DEBUG_MODE
        print(f"Debug mode {'enabled' if DEBUG_MODE else 'disabled'}")
        if not DEBUG_MODE:
            cv2.destroyWindow("Motion Detection")
            cv2.destroyWindow("Static Detection") 
            cv2.destroyWindow("Combined Detection")

cap.release()
cv2.destroyAllWindows()