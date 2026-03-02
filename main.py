import cv2
import easyocr
import re
from ultralytics import YOLO
from database import ALPRDatabase

def clean_and_validate_plate(raw_text):
    """
    Cleans OCR text and validates based on official WA State rules:
    - 1 to 7 characters long.
    - Allows Letters, Numbers, Hyphens, and Spaces.
    """
    # 1. Keep ONLY letters, numbers, hyphens, and spaces
    cleaned = re.sub(r'[^A-Z0-9\-\s]', '', raw_text.upper())
    
    # 2. Strip leading and trailing spaces or hyphens
    cleaned = cleaned.strip(' -')
    
    # 3. Filter out known "Stop Words"
    stop_words = ["WASHINGTON", "STATE", "EVERGREEN", "WASH", "GTON", "TOIN", "WA"]
    if cleaned in stop_words:
        return None
        
    # 4. Filter by official length (1 to 7 characters)
    if len(cleaned) < 1 or len(cleaned) > 7:
        return None
        
    # 5. Require at least one alphanumeric character
    if not any(char.isalnum() for char in cleaned):
        return None
        
    # Optional: Strip spaces and hyphens for a cleaner database entry
    cleaned = cleaned.replace(" ", "").replace("-", "")
        
    return cleaned

def main():
    # 1. Initialize DB and Models
    db = ALPRDatabase(db_path="plates.db")
    
    # Load your YOLO model (Make sure this matches your actual .pt file name!)
    model = YOLO("license_plate_detector.pt") 
    
    # Initialize OCR
    reader = easyocr.Reader(['en'], gpu=False) # gpu=False since your log showed CPU usage

    cap = cv2.VideoCapture(0)
    print("ALPR System Active. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 2. Run YOLO Inference
        results = model.predict(source=frame, conf=0.5, imgsz=640, verbose=False)

        for result in results:
            for box in result.boxes:
                # Get bounding box coordinates for the plate
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_crop = frame[y1:y2, x1:x2]

                if plate_crop.size > 0:
                    # 3. OCR on the crop
                    ocr_res = reader.readtext(plate_crop)
                    
                    if ocr_res:
                        largest_area = 0
                        best_text_candidate = ""
                        best_confidence = 0.0

                        # 4. Find the physically largest text (ignores tiny state names)
                        for detection in ocr_res:
                            bbox = detection[0]
                            raw_text = detection[1]
                            conf = detection[2]

                            # Calculate width and height to find the area
                            width = bbox[2][0] - bbox[0][0]
                            height = bbox[2][1] - bbox[0][1]
                            area = width * height

                            if area > largest_area:
                                largest_area = area
                                best_text_candidate = raw_text
                                best_confidence = conf

                        # 5. Process ONLY the largest text candidate
                        if best_confidence >= 0.4:
                            validated_plate = clean_and_validate_plate(best_text_candidate)
                            
                            if validated_plate:
                                # Log it to the database
                                db.log_detection(validated_plate, best_confidence, plate_crop)

                                # Draw visualization on the live feed
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, f"{validated_plate}", (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("ALPR - YOLO + EasyOCR", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()