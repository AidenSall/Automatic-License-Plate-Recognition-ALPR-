#test duplicate of database.py to test saving function 


import sqlite3
import datetime
import os
import cv2
import numpy as np
import time

# --- REVISED DATABASE CLASS ---
class ALPRDatabase:
    def __init__(self, db_path):
        self.db_path = db_path
        self.img_dir = os.path.join(os.path.dirname(self.db_path), "crops")
        self._initialize_db()

    def _initialize_db(self):
        # Ensure the directory for the DB file exists
        db_folder = os.path.dirname(self.db_path)
        if not os.path.exists(db_folder):
            os.makedirs(db_folder)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('PRAGMA journal_mode = WAL;')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    plate_text TEXT NOT NULL,
                    confidence REAL,
                    image_path TEXT
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_plate ON detections(plate_text)')
            conn.commit()

    def log_detection(self, plate_text, confidence, frame_crop):
        plate_text = plate_text.upper().strip()
        
        # --- LOGIC UPDATE: DE-DUPLICATION ---
        # Check if this plate was seen in the last 5 seconds
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT timestamp FROM detections 
                WHERE plate_text = ? 
                ORDER BY id DESC LIMIT 1
            ''', (plate_text,))
            last_entry = cursor.fetchone()

        if last_entry:
            last_seen = datetime.datetime.strptime(last_entry[0], "%Y-%m-%d %H:%M:%S.%f")
            time_diff = (datetime.datetime.now() - last_seen).total_seconds()
            
            if time_diff < 5:
                print(f"[SKIP] Duplicate: {plate_text} (Seen {time_diff:.2f}s ago)")
                return # Exit function, do not save

        # --- SAVE IMAGE ---
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        now = datetime.datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S.%f")
        img_filename = f"plate_{now.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        full_img_path = os.path.join(self.img_dir, img_filename)

        success = cv2.imwrite(full_img_path, frame_crop)
        if not success:
            print(f"[ERR] Failed to save image: {full_img_path}")
            return

        # --- SAVE TO DB ---
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO detections (timestamp, plate_text, confidence, image_path)
                    VALUES (?, ?, ?, ?)
                ''', (timestamp_str, plate_text, round(confidence, 4), img_filename))
                conn.commit()
                print(f"[LOG] Saved new detection: {plate_text}")
        except sqlite3.Error as e:
            print(f"[ERR] Database insertion error: {e}")

# --- TEST HARNESS ---

def create_dummy_image(text):
    """Generates a black image with the license plate text written on it."""
    img = np.zeros((100, 300, 3), dtype=np.uint8)
    cv2.putText(img, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                1.5, (255, 255, 255), 2, cv2.LINE_AA)
    return img

def main():
    # 1. Setup local test path (avoiding /mnt/sdcard for desktop testing)
    test_db_path = "./alpr_test_data/plates.db"
    
    # Remove old test data to start fresh
    if os.path.exists(test_db_path):
        try:
            os.remove(test_db_path)
            print("Cleaned up old database.")
        except PermissionError:
            print("Could not delete old DB (file might be open).")
            
    print(f"Initializing Database at {test_db_path}...")
    db = ALPRDatabase(db_path=test_db_path)

    # 2. Simulate "Plate Chatter" (Rapid fire detections of same car)
    print("\n--- TEST 1: Rapid Duplicate Detection (Car 'ABC-123') ---")
    plate_a = "ABC-123"
    dummy_img_a = create_dummy_image(plate_a)

    for i in range(5):
        # Sending 5 detections in a row with slight variations in confidence
        conf = 0.90 + (i * 0.01) 
        print(f"Frame {i+1}: sending {plate_a}...", end=" ")
        db.log_detection(plate_a, conf, dummy_img_a)
        time.sleep(0.1) # Simulate 100ms between frames

    # 3. Simulate a new car arriving
    print("\n--- TEST 2: New Car Arrives ('XYZ-999') ---")
    plate_b = "XYZ-999"
    dummy_img_b = create_dummy_image(plate_b)
    db.log_detection(plate_b, 0.95, dummy_img_b)

    # 4. Simulate the first car returning after the cooldown
    print("\n--- TEST 3: Original Car Returns (Simulated delay) ---")
    # We can't actually wait 5 seconds in a quick test, so we manually
    # hack the DB timestamp for the test or just wait. Let's just wait 5s for realism.
    print("Waiting 5.1 seconds for cooldown to expire...")
    time.sleep(5.1)
    
    print(f"Sending {plate_a} again...")
    db.log_detection(plate_a, 0.92, dummy_img_a)

    # 5. Verify contents
    print("\n--- VERIFICATION ---")
    conn = sqlite3.connect(test_db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, timestamp, plate_text, image_path FROM detections")
    rows = cursor.fetchall()
    
    print(f"Total Rows in DB: {len(rows)}")
    print(f"{'ID':<4} {'Timestamp':<25} {'Plate':<10} {'Image File'}")
    print("-" * 60)
    for row in rows:
        print(f"{row[0]:<4} {row[1]:<25} {row[2]:<10} {row[3]}")
    
    conn.close()
    print("\nTest Complete. Check './alpr_test_data/crops' to see saved images.")

if __name__ == "__main__":
    main()