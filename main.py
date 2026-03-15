import cv2
import pytesseract
from pytesseract import Output
import re
import time
import numpy as np
import onnxruntime as ort
from database import ALPRDatabase
import os

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

def save_failed_read(original_crop, thresh_crop, raw_text, reason, conf=0.0):
    """
    Saves the original and thresholded crops side-by-side to disk.
    Filename includes the reason for failure, confidence, and raw text guess.
    """
    save_dir = "failed_reads"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # Sanitize and truncate the text to prevent OS filename errors from garbage reads
    safe_text = re.sub(r'[^A-Za-z0-9]', '_', raw_text)[:15]
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    filename = f"{reason}_{timestamp}_{safe_text}_c{int(conf*100)}.jpg"
    filepath = os.path.join(save_dir, filename)
    
    # Convert the grayscale threshold back to 3-channel BGR for concatenation
    thresh_bgr = cv2.cvtColor(thresh_crop, cv2.COLOR_GRAY2BGR)
    
    # Resize just in case to prevent OpenCV dimension mismatch errors
    h, w = original_crop.shape[:2]
    thresh_bgr = cv2.resize(thresh_bgr, (w, h))
        
    # Stitch them horizontally (Original on left, Threshold on right)
    debug_img = cv2.hconcat([original_crop, thresh_bgr])
    cv2.imwrite(filepath, debug_img)

def print_performance_metrics(captured, processed, elapsed):
    """Prints formatted performance metrics."""
    fps_captured = captured / elapsed
    fps_processed = processed / elapsed
    print("\n" + "="*50)
    print(f" METRICS OVER LAST {elapsed:.1f} SECONDS")
    print(f" Frames Captured:     {captured} ({fps_captured:.2f} FPS)")
    print(f" Frames Processed:    {processed} ({fps_processed:.2f} FPS)")
    print("="*50 + "\n")

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

    # 1. Define the GStreamer pipeline
    gst_pipeline = (
        "libcamerasrc awb-mode=auto ! "
        "video/x-raw, width=640, height=480, framerate=15/1 ! "
        "videoconvert ! "
        "videoscale ! "
        "video/x-raw, width=640, height=480, format=BGR ! "
        "appsink drop=True"
    )

    # 2. Initialize the Capture
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("ERROR: GStreamer pipeline failed to open.")
        print("Check if another process (like rpicam-hello) is using the camera.")
        exit()
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("-" * 40)
    print(f"ALPR System Active. Model Size: {input_width}x{input_height}")
    print("Press 'q' in the video window or Ctrl+C in terminal to quit.")
    print("-" * 40)

    frame_skip = 3 
    total_frames_captured = 0
    total_frames_processed = 0

    start_time = time.time()
    evaluation_window = 60.0 

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                print("Warning: Failed to grab frame. Retrying...")
                time.sleep(1)
                continue
            
            # Reset thresh_plate at the start of every frame to avoid NameError
            thresh_plate = None

            total_frames_captured += 1
            
            if total_frames_captured % frame_skip != 0:
                elapsed_time = time.time() - start_time
                if elapsed_time >= evaluation_window:
                    print_performance_metrics(total_frames_captured, total_frames_processed, elapsed_time)
                    start_time = time.time()
                    total_frames_captured = 0
                    total_frames_processed = 0
                
                # Render the live feed even on skipped frames
                cv2.imshow("ALPR Live Feed", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            total_frames_processed += 1

            # --- PRE-PROCESS IMAGE FOR ONNX ---
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (input_width, input_height))
            img_data = np.array(img).astype(np.float32) / 255.0
            img_data = np.transpose(img_data, (2, 0, 1)) 
            img_data = np.expand_dims(img_data, axis=0)  

            # --- RUN INFERENCE ---
            outputs = session.run(None, {input_name: img_data})
            predictions = np.squeeze(outputs[0]).T 
            
            original_height, original_width = frame.shape[:2]
            
            model_w, model_h = 640, 640
            x_factor = original_width / model_w
            y_factor = original_height / model_h
            
            boxes = []
            confidences = []
            
            conf_threshold = 0.25
            valid_predictions = predictions[predictions[:, 4] > conf_threshold]
            
            for pred in valid_predictions:
                x_c, y_c, w, h, conf = pred
                
                if w <= 1.5 and h <= 1.5:
                    x_c *= original_width
                    y_c *= original_height
                    w *= original_width
                    h *= original_height
                else:
                    x_c *= x_factor
                    y_c *= y_factor
                    w *= x_factor
                    h *= y_factor
                
                x_min = int(x_c - (w / 2))
                y_min = int(y_c - (h / 2))
                
                boxes.append([x_min, y_min, int(w), int(h)])
                confidences.append(float(conf))
                
            # --- NON-MAXIMUM SUPPRESSION ---
            indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.2, nms_threshold=0.4)
            
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    
                    x = max(0, x)
                    y = max(0, y)
                    w = min(original_width - x, w)
                    h = min(original_height - y, h)
                    
                    plate_crop = frame[y:y+h, x:x+w]

                    if plate_crop.size > 0:
                        # --- TESSERACT OCR PIPELINE ---
                        # 1. Convert to Grayscale
                        gray_plate = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                        
                        # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        enhanced_gray = clahe.apply(gray_plate)
                        
                        # 3. Bilateral Filter
                        blurred = cv2.bilateralFilter(enhanced_gray, d=11, sigmaColor=17, sigmaSpace=17)
                        
                        # 4. Adaptive Thresholding
                        thresh_plate_temp = cv2.adaptiveThreshold(
                            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                            cv2.THRESH_BINARY_INV, 19, 10
                        )
                        
                        # 5. Morphological Opening (Noise removal)
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                        clean_thresh = cv2.morphologyEx(thresh_plate_temp, cv2.MORPH_OPEN, kernel)
                        
                        # 6. Morphological Dilation (Stroke repair)
                        repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                        clean_thresh = cv2.dilate(clean_thresh, repair_kernel, iterations=1)
                        
                        # 7. Invert back to black text on white background for Tesseract
                        final_thresh = cv2.bitwise_not(clean_thresh)
                        
                        # Map to the variable used by visual debugging and failed read saves
                        thresh_plate = final_thresh
                        
                        # 8. Tesseract Configuration
                        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
                        ocr_data = pytesseract.image_to_data(final_thresh, config=custom_config, output_type=Output.DICT)
                        
                        best_text_candidate = ""
                        best_confidence = 0.0

                        for j in range(len(ocr_data['text'])):
                            conf = float(ocr_data['conf'][j]) / 100.0
                            text = ocr_data['text'][j].strip()
                            
                            if conf > best_confidence and len(text) > 0:
                                best_confidence = conf
                                best_text_candidate = text

                        # --- LOGIC GATES ---
                        if best_text_candidate:
                            if best_confidence >= 0.4:
                                validated_plate = clean_and_validate_plate(best_text_candidate)
                                
                                if validated_plate:
                                    timestamp = time.strftime('%H:%M:%S')
                                    print(f"[{timestamp}] Found: {validated_plate} ({best_confidence*100:.1f}%)")
                                    db.log_detection(validated_plate, best_confidence, plate_crop)
                                else:
                                    print(f"[-] Dropped (Validation): '{best_text_candidate}' failed WA state syntax rules.")
                                    save_failed_read(plate_crop, thresh_plate, best_text_candidate, "VAL", best_confidence)
                            else:
                                print(f"[-] Dropped (Confidence): '{best_text_candidate}' was only {best_confidence*100:.1f}% confident.")
                                save_failed_read(plate_crop, thresh_plate, best_text_candidate, "CONF", best_confidence)
            
            # --- METRICS EVALUATION ---
            elapsed_time = time.time() - start_time
            if elapsed_time >= evaluation_window:
                print_performance_metrics(total_frames_captured, total_frames_processed, elapsed_time)
                start_time = time.time()
                total_frames_captured = 0
                total_frames_processed = 0

            # --- VISUAL DEBUGGING ---
            # Render the threshold crop only if a plate was actually processed
            if thresh_plate is not None:
                cv2.imshow("Tesseract Vision (Thresh)", thresh_plate)
            
            cv2.imshow("ALPR Live Feed", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Shutting down gracefully...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released. System offline.")

if __name__ == "__main__":
    main()