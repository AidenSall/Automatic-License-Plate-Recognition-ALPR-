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
        Saves the image crop to disk and logs the metadata to SQLite.
        :param plate_text: String result from the OCR engine.
        :param confidence: The confidence score from the OCR engine.
        :param frame_crop: The OpenCV image (numpy array) containing the license plate.
        """
        # Ensure the storage directory exists
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        # Generate a unique filename using microseconds to avoid collisions 
        # during high-speed traffic (multiple cars per second).
        now = datetime.datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S.%f")
        img_filename = f"plate_{now.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        full_img_path = os.path.join(self.img_dir, img_filename)

        # Write the image to the SD card using OpenCV
        success = cv2.imwrite(full_img_path, frame_crop)
        if not success:
            print(f"Error: Could not write image to {full_img_path}")
            return

        # Insert metadata into the database
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Use parameterized queries (?) to prevent SQL injection and handle formatting
                cursor.execute('''
                    INSERT INTO detections (timestamp, plate_text, confidence, image_path)
                    VALUES (?, ?, ?, ?)
                ''', (timestamp_str, plate_text.upper().strip(), round(confidence, 4), img_filename))
                conn.commit()
        except sqlite3.Error as e:
            # Crucial for debugging: SQLite may fail if the SD card is full or read-only
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