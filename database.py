import sqlite3
import datetime
import os
import cv2

class ALPRDatabase:
    """
    Handles persistent storage for the ALPR system, managing both a 
    SQLite database for metadata and local disk storage for image crops.
    """

    def __init__(self, db_path="/mnt/sdcard/alpr_data/plates.db"):
        """
        Initializes the database handler.
        :param db_path: Absolute path to the .db file on the SD card.
        """
        self.db_path = db_path
        # Define the directory for images relative to the database file
        self.img_dir = os.path.join(os.path.dirname(self.db_path), "crops")
        self._initialize_db()

    def _initialize_db(self):
        """
        Sets up the database structure and optimizes settings for SD card use.
        Uses a context manager (with) to ensure the connection closes automatically.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 1. Performance Tuning: Write-Ahead Logging (WAL)
            # This reduces disk I/O, which extends the life of the SD card.
            cursor.execute('PRAGMA journal_mode = WAL;')
            
            # 2. Table Creation
            # id: Auto-incrementing primary key for unique row identification
            # timestamp: Uses ISO 8601 format for easy sorting/filtering
            # plate_text: The OCR result
            # confidence: Float representing the OCR certainty (0.0 to 1.0)
            # image_path: Local filename of the saved crop for visual audit
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    plate_text TEXT NOT NULL,
                    confidence REAL,
                    image_path TEXT
                )
            ''')
            
            # 3. Indexing
            # Speeds up queries like "Find all records for plate X" as the DB grows.
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_plate ON detections(plate_text)')
            conn.commit()

    def log_detection(self, plate_text, confidence, frame_crop):
        """
        Logs detection ONLY if the plate hasn't been seen recently.
        """
        plate_text = plate_text.upper().strip()
        
        # 1. Check for duplicates (De-duplication logic)
        # We query the DB for the most recent sighting of THIS plate
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT timestamp FROM detections 
                WHERE plate_text = ? 
                ORDER BY id DESC LIMIT 1
            ''', (plate_text,))
            last_entry = cursor.fetchone()

        if last_entry:
            # Parse the timestamp string back to a datetime object
            # Format must match the one used in insertion
            last_seen = datetime.datetime.strptime(last_entry[0], "%Y-%m-%d %H:%M:%S.%f")
            time_diff = (datetime.datetime.now() - last_seen).total_seconds()
            
            # CONSTRAINT: If seen within the last 5 seconds, ignore it.
            if time_diff < 5:
                print(f"Skipping duplicate: {plate_text} (seen {time_diff:.1f}s ago)")
                return

        # 2. Proceed with storage if it's a new or "old enough" detection
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        now = datetime.datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S.%f")
        img_filename = f"plate_{now.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        full_img_path = os.path.join(self.img_dir, img_filename)

        success = cv2.imwrite(full_img_path, frame_crop)
        if not success:
            print(f"Error: Could not write image to {full_img_path}")
            return

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO detections (timestamp, plate_text, confidence, image_path)
                    VALUES (?, ?, ?, ?)
                ''', (timestamp_str, plate_text, round(confidence, 4), img_filename))
                conn.commit()
                print(f"Logged: {plate_text}")
        except sqlite3.Error as e:
            print(f"Database insertion error: {e}")

    def get_plate_history(self, plate_text):
        """
        Retrieves all historical sightings of a specific license plate.
        :param plate_text: The string to search for.
        :return: A list of tuples containing (timestamp, confidence).
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            query = "SELECT timestamp, confidence FROM detections WHERE plate_text = ? ORDER BY timestamp DESC"
            cursor.execute(query, (plate_text.upper().strip(),))
            return cursor.fetchall()