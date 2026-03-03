import cv2
import easyocr
import re
import time
from ultralytics import YOLO
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
    model = YOLO("your_model.pt") 
    reader = easyocr.Reader(['en'], gpu=False)

    cap = cv2.VideoCapture(0)
    
    # Pi Optimization 1: Force lower resolution for faster processing
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("-" * 40)
    print("Headless ALPR System Active.")
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

            # YOLO Inference (verbose=False keeps the terminal clean)
            results = model.predict(source=frame, conf=0.5, imgsz=640, verbose=False)

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    plate_crop = frame[y1:y2, x1:x2]

                    if plate_crop.size > 0:
                        # EasyOCR
                        ocr_res = reader.readtext(plate_crop)
                        
                        if ocr_res:
                            largest_area = 0
                            best_text_candidate = ""
                            best_confidence = 0.0

                            # Find the physically largest text
                            for detection in ocr_res:
                                bbox = detection[0]
                                raw_text = detection[1]
                                conf = detection[2]

                                width = bbox[2][0] - bbox[0][0]
                                height = bbox[2][1] - bbox[0][1]
                                area = width * height

                                if area > largest_area:
                                    largest_area = area
                                    best_text_candidate = raw_text
                                    best_confidence = conf

                            # Validate and Log
                            if best_confidence >= 0.4:
                                validated_plate = clean_and_validate_plate(best_text_candidate)
                                
                                if validated_plate:
                                    # Print to console for easy monitoring
                                    timestamp = time.strftime('%H:%M:%S')
                                    print(f"[{timestamp}] Found: {validated_plate} ({best_confidence*100:.1f}%)")
                                    
                                    # Send to SQLite and save crop
                                    db.log_detection(validated_plate, best_confidence, plate_crop)

    # Pi Optimization 3: Graceful Exit without waitKey()
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Shutting down gracefully...")
    finally:
        cap.release()
        print("Camera released. Goodbye!")

if __name__ == "__main__":
    main()