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
    cleaned = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
    
    # Fallback stop-word removal just in case the crop caught the bottom edge of a word
    stop_words = ["WASHINGTON", "STATE", "EVERGREEN", "WASH", "GTON", "TOIN", "WA"]
    for word in stop_words:
        cleaned = cleaned.replace(word, "")
        
    if len(cleaned) < 1 or len(cleaned) > 7:
        return None
        
    return cleaned

def save_failed_read(original_crop, thresh_crop, raw_text, reason, conf=0.0):
    save_dir = "failed_reads"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    safe_text = re.sub(r'[^A-Za-z0-9]', '_', raw_text)[:15]
    if not safe_text:
        safe_text = "EMPTY"
        
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filename = f"{reason}_{timestamp}_{safe_text}_c{int(conf*100)}.jpg"
    filepath = os.path.join(save_dir, filename)
    
    thresh_bgr = cv2.cvtColor(thresh_crop, cv2.COLOR_GRAY2BGR)
    h, w = original_crop.shape[:2]
    thresh_bgr = cv2.resize(thresh_bgr, (w, h))
        
    debug_img = cv2.hconcat([original_crop, thresh_bgr])
    cv2.imwrite(filepath, debug_img)

def print_performance_metrics(captured, processed, elapsed):
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
    
    model_path = "license_plate_detector.onnx"
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    model_inputs = session.get_inputs()
    input_name = model_inputs[0].name
    input_shape = model_inputs[0].shape
    input_width = input_shape[3]
    input_height = input_shape[2]

    gst_pipeline = (
        "libcamerasrc awb-mode=auto ! "
        "video/x-raw, width=640, height=480, framerate=15/1 ! "
        "videoconvert ! "
        "videoscale ! "
        "video/x-raw, width=640, height=480, format=BGR ! "
        "appsink drop=True"
    )

    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("ERROR: GStreamer pipeline failed to open.")
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
    
    session_ocr_log = []

    start_time = time.time()
    evaluation_window = 60.0 

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                time.sleep(1)
                continue
            
            thresh_plate = None
            total_frames_captured += 1
            
            if total_frames_captured % frame_skip != 0:
                elapsed_time = time.time() - start_time
                if elapsed_time >= evaluation_window:
                    print_performance_metrics(total_frames_captured, total_frames_processed, elapsed_time)
                    start_time = time.time()
                    total_frames_captured = 0
                    total_frames_processed = 0
                
                cv2.imshow("ALPR Live Feed", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            total_frames_processed += 1

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (input_width, input_height))
            img_data = np.array(img).astype(np.float32) / 255.0
            img_data = np.transpose(img_data, (2, 0, 1)) 
            img_data = np.expand_dims(img_data, axis=0)  

            outputs = session.run(None, {input_name: img_data})
            predictions = np.squeeze(outputs[0]).T 
            
            original_height, original_width = frame.shape[:2]
            x_factor = original_width / 640
            y_factor = original_height / 640
            
            boxes = []
            confidences = []
            
            valid_predictions = predictions[predictions[:, 4] > 0.25]
            
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
                
            indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.2, nms_threshold=0.4)
            
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    x, y = max(0, x), max(0, y)
                    w, h = min(original_width - x, w), min(original_height - y, h)
                    
                    plate_crop = frame[y:y+h, x:x+w]

                    if plate_crop.size > 0:
                        # --- HEURISTIC VERTICAL CROP ---
                        # Shave 20% off the top (State name/Stickers) and 15% off the bottom (Motto)
                        crop_h, crop_w = plate_crop.shape[:2]
                        top_trim = int(crop_h * 0.20)
                        bottom_trim = int(crop_h * 0.15)
                        
                        # Ensure we don't accidentally invert the crop on tiny boxes
                        if crop_h > (top_trim + bottom_trim):
                            core_plate_crop = plate_crop[top_trim:crop_h-bottom_trim, :]
                        else:
                            core_plate_crop = plate_crop # Fallback if bounding box is extremely skewed
                            
                        # --- TESSERACT OCR PIPELINE ---
                        gray_plate = cv2.cvtColor(core_plate_crop, cv2.COLOR_BGR2GRAY)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        enhanced_gray = clahe.apply(gray_plate)
                        blurred = cv2.bilateralFilter(enhanced_gray, d=11, sigmaColor=17, sigmaSpace=17)
                        
                        thresh_plate_temp = cv2.adaptiveThreshold(
                            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                            cv2.THRESH_BINARY_INV, 19, 10
                        )
                        
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                        clean_thresh = cv2.morphologyEx(thresh_plate_temp, cv2.MORPH_OPEN, kernel)
                        repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                        clean_thresh = cv2.dilate(clean_thresh, repair_kernel, iterations=1)
                        
                        final_thresh = cv2.bitwise_not(clean_thresh)
                        thresh_plate = final_thresh
                        
                        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
                        ocr_data = pytesseract.image_to_data(final_thresh, config=custom_config, output_type=Output.DICT)
                        
                        raw_ocr_words = [t.strip() for t in ocr_data['text'] if t.strip()]
                        raw_full_string = " ".join(raw_ocr_words)
                        
                        # --- CONCATENATED EXTRACTION LOGIC ---
                        best_text_candidate = ""
                        total_conf = 0.0
                        valid_char_count = 0

                        for j in range(len(ocr_data['text'])):
                            text = ocr_data['text'][j].strip()
                            conf = float(ocr_data['conf'][j]) / 100.0
                            
                            if len(text) > 0 and conf > 0.0:
                                clean_chunk = re.sub(r'[^A-Z0-9]', '', text.upper())
                                if clean_chunk:
                                    best_text_candidate += clean_chunk
                                    # Weight the confidence by the length of the string chunk
                                    total_conf += (conf * len(clean_chunk)) 
                                    valid_char_count += len(clean_chunk)
                                    
                        # Calculate weighted average confidence across the whole plate
                        best_confidence = (total_conf / valid_char_count) if valid_char_count > 0 else 0.0

                        # --- LOGIC GATES WITH LOGGING ---
                        status_msg = ""
                        if best_text_candidate:
                            if best_confidence >= 0.4:
                                validated_plate = clean_and_validate_plate(best_text_candidate)
                                if validated_plate:
                                    status_msg = "SUCCESS"
                                    timestamp = time.strftime('%H:%M:%S')
                                    print(f"[{timestamp}] Found: {validated_plate} ({best_confidence*100:.1f}%)")
                                    db.log_detection(validated_plate, best_confidence, core_plate_crop)
                                else:
                                    status_msg = "FAILED_VALIDATION (Syntax/Stop-word/Length)"
                                    save_failed_read(core_plate_crop, thresh_plate, best_text_candidate, "VAL", best_confidence)
                            else:
                                status_msg = "FAILED_CONFIDENCE (< 40%)"
                                save_failed_read(core_plate_crop, thresh_plate, best_text_candidate, "CONF", best_confidence)
                        else:
                            status_msg = "FAILED_NO_TEXT_FOUND (Tesseract returned empty)"
                            best_text_candidate = "NONE"

                        log_timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                        log_entry = f"[{log_timestamp}] RAW: '{raw_full_string}' | TARGET: '{best_text_candidate}' | CONF: {best_confidence*100:.1f}% | STATUS: {status_msg}"
                        session_ocr_log.append(log_entry)
                        
                        if status_msg != "SUCCESS":
                            print(f"[-] Dropped: '{best_text_candidate}' | Reason: {status_msg}")
            
            elapsed_time = time.time() - start_time
            if elapsed_time >= evaluation_window:
                print_performance_metrics(total_frames_captured, total_frames_processed, elapsed_time)
                start_time = time.time()
                total_frames_captured = 0
                total_frames_processed = 0

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
        
        # Write the session log to disk on exit
        if session_ocr_log:
            log_filename = "ocr_session_log.txt"
            with open(log_filename, "w") as f:
                f.write("=== ALPR OCR SESSION LOG ===\n")
                f.write("\n".join(session_ocr_log))
            print(f"Saved {len(session_ocr_log)} OCR events to {log_filename}.")
            
        print("Camera released. System offline.")

if __name__ == "__main__":
    main()