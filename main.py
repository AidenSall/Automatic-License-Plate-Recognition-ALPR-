import cv2
import pytesseract
from pytesseract import Output
import re
import time
import numpy as np
import onnxruntime as ort
from database import ALPRDatabase

def clean_and_validate_plate(raw_text):
    """
    Cleans OCR text and validates based on official WA State rules.
    """
    cleaned = re.sub(r'[^A-Z0-9\-\s]', '', raw_text.upper())
    cleaned = cleaned.strip(' -')
    
    stop_words = ["WASHINGTON", "STATE", "EVERGREEN", "WASH", "GTON", "TOIN", "WA"]
    if cleaned in stop_words:
        return None
        
    if len(cleaned) < 1 or len(cleaned) > 7:
        return None
        
    if not any(char.isalnum() for char in cleaned):
        return None
        
    cleaned = cleaned.replace(" ", "").replace("-", "")
    return cleaned

def main():
    print("Initializing Database and Models... (This takes a moment on a Raspberry Pi)")
    db = ALPRDatabase(db_path="plates.db")
    
    # Initialize ONNX Runtime (Bypasses PyTorch entirely)
    model_path = "license_plate_detector.onnx"
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    # Extract dynamic input shapes expected by the model
    model_inputs = session.get_inputs()
    input_name = model_inputs[0].name
    input_shape = model_inputs[0].shape
    input_width = input_shape[3]
    input_height = input_shape[2]

    # 1. Define the GStreamer pipeline for the Pi 3 on Trixie
    # We use 15fps to keep the CPU cool while running YOLO
    gst_pipeline = (
        "libcamerasrc ! "
        "video/x-raw, width=640, height=480, framerate=15/1 ! "
        "videoconvert ! "
        "videoscale ! "
        "video/x-raw, width=640, height=480, format=BGR ! "
        "appsink drop=True"
    )

    # 2. Initialize the Capture
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

    # 3. Critical Check
    if not cap.isOpened():
        print("ERROR: GStreamer pipeline failed to open.")
        print("Check if another process (like rpicam-hello) is using the camera.")
        exit()
    
    # Pi Optimization 1: Force lower resolution for faster processing
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("-" * 40)
    print(f"Headless ALPR System Active. Model Size: {input_width}x{input_height}")
    print("Press Ctrl+C to quit.")
    print("-" * 40)

    # Pi Optimization 2: Frame Skipping
    frame_skip = 3  # Process 1 out of every 3 frames
    frame_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                print("Warning: Failed to grab frame. Retrying...")
                time.sleep(1)
                continue

            frame_count += 1
            
            # Skip frames to give the Raspberry Pi CPU time to breathe
            if frame_count % frame_skip != 0:
                continue

            # --- PRE-PROCESS IMAGE FOR ONNX ---
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (input_width, input_height))
            img_data = np.array(img).astype(np.float32) / 255.0
            img_data = np.transpose(img_data, (2, 0, 1)) 
            img_data = np.expand_dims(img_data, axis=0)  

            # --- RUN INFERENCE ---
            outputs = session.run(None, {input_name: img_data})
            
            # --- POST-PROCESS OUTPUTS ---
            predictions = np.squeeze(outputs[0]).T 
            
            original_height, original_width = frame.shape[:2]
            x_factor = original_width / input_width
            y_factor = original_height / input_height
            
            boxes = []
            confidences = []
            
            # Filter low confidence before applying scaling math
            conf_threshold = 0.5
            valid_predictions = predictions[predictions[:, 4] > conf_threshold]
            
            for pred in valid_predictions:
                x_c, y_c, w, h, conf = pred
                
                # Scale coordinates back to actual camera resolution
                x_c *= x_factor
                y_c *= y_factor
                w *= x_factor
                h *= y_factor
                
                # Convert center to top-left
                x_min = int(x_c - (w / 2))
                y_min = int(y_c - (h / 2))
                
                boxes.append([x_min, y_min, int(w), int(h)])
                confidences.append(float(conf))
                
            # --- NON-MAXIMUM SUPPRESSION (Remove Overlaps) ---
            # Temporarily lowered threshold to 0.2 to force the model to show its guesses
            indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.2, nms_threshold=0.4)
            
            if len(indices) > 0:
                print(f"YOLO saw {len(indices)} potential objects...") # Debug print
                
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    
                    # Prevent array out-of-bounds on edge cases
                    x = max(0, x)
                    y = max(0, y)
                    w = min(original_width - x, w)
                    h = min(original_height - y, h)
                    
                    # --- DRAW THE BOUNDING BOX ---
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"Plate: {confidences[i]:.2f}", (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    plate_crop = frame[y:y+h, x:x+w]

                    if plate_crop.size > 0:
                        # --- TESSERACT OCR PIPELINE ---
                        # 1. Pre-process the crop for Tesseract (Grayscale & Threshold)
                        gray_plate = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                        _, thresh_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                        
                        # 2. Configure Tesseract to look for a single line of alphanumeric text
                        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
                        
                        # 3. Read the text and confidence scores
                        ocr_data = pytesseract.image_to_data(thresh_plate, config=custom_config, output_type=Output.DICT)
                        
                        best_text_candidate = ""
                        best_confidence = 0.0

                        # Find the highest confidence word detected
                        for i in range(len(ocr_data['text'])):
                            conf = float(ocr_data['conf'][i]) / 100.0  # Tesseract returns 0-100, we need 0.0-1.0
                            text = ocr_data['text'][i].strip()
                            
                            if conf > best_confidence and len(text) > 0:
                                best_confidence = conf
                                best_text_candidate = text

                        # Validate and Log
                        if best_confidence >= 0.4:
                            validated_plate = clean_and_validate_plate(best_text_candidate)
                            
                            if validated_plate:
                                timestamp = time.strftime('%H:%M:%S')
                                print(f"[{timestamp}] Found: {validated_plate} ({best_confidence*100:.1f}%)")
                                db.log_detection(validated_plate, best_confidence, plate_crop)
# --- VISUAL DEBUGGING ---
            # Show the live camera feed in a window
            cv2.imshow("ALPR Live Feed", frame)
            
            # Press 'q' on your keyboard to quit the window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Shutting down gracefully...")
    finally:
        cap.release()
        print("Camera released. Goodbye!")

if __name__ == "__main__":
    main()