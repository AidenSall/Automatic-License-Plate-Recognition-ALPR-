# Automatic-License-Plate-Recognition-ALPR-
This project develops a real-time mobile Automatic License Plate Recognition (ALPR) prototype for non-stationary platforms. It integrates a hardware-software stack to capture high-speed images, isolate license plates, and extract alphanumeric data from surrounding traffic.




To Do:

1. The "Plate Chatter" ProblemALPR engines typically detect the same license plate in multiple consecutive frames (e.g., 10 detections for a single car passing by). Your current log_detection method blindly inserts every single detection.Consequence: Your database will be flooded with near-identical entries, and your SD card usage will spike unnecessarily.Fix: Implement a "cooldown" period. Only log a plate if it hasn't been seen in the last $X$ seconds.


2. Blocking I/O BottlenecksYou are running cv2.imwrite (image saving) and sqlite

3.connect inside the main execution flow. Writing to an SD card is slow.Consequence: While the system saves the image, your camera frame acquisition will freeze. If writing takes 200ms, your frame rate drops to 5 FPS, causing you to miss fast-moving cars.Fix (For now): Accept the lag for initial testing.Fix (Production): Move log_detection to a separate background thread or a queue so the main loop keeps running.3. Connection OverheadOpening and closing the SQLite connection (with sqlite3.connect...) for every detection is inefficient on embedded systems.Consequence: High CPU overhead and increased latency during writes.Fix: Keep the connection open as a class attribute (self.conn) and close it only when the program terminates.
