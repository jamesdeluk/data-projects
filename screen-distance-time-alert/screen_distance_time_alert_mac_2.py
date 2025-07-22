import rumps
import cv2
from cvzone.FaceMeshModule import FaceMeshDetector
import time
import collections

class ScreenDistanceApp(rumps.App):
    def __init__(self):
        super().__init__("Screen Distance", icon=None)
        self.title = "Starting..."
        
        # Face detection setup
        self.cap = cv2.VideoCapture(0)
        self.detector = FaceMeshDetector(maxFaces=1)
        
        # Measurement variables
        self.distance_history = collections.deque(maxlen=100)
        
        self.face_time_threshold = 30 /60  # Seconds in minutes

        # Time tracking for different states
        self.face_time = 0  # Time with face detected
        self.no_face_time = 0  # Time with no face detected
        self.face_time_count = 0  # Time with face detected
        self.no_face_time_count = 0  # Time with no face detected
        self.state_change_time = self.face_time_threshold  # Ensure passes the threshold first run
        self.last_time = time.time()
        self.face_detected = False  # Tracks the last state
        self.ratio = 0
        self.avg_distance = 0

        # Timer to update the menu bar title
        self.timer = rumps.Timer(self.update_distance, 1)  # Update every second
        self.timer.start()

    def update_distance(self, _):
        success, img = self.cap.read()
        img, faces = self.detector.findFaceMesh(img, draw=False)
        
        current_time = time.time()
        elapsed = (current_time - self.last_time) / 60
        self.last_time = current_time  # Update for next iteration

        if faces:
            self.state_change_time += elapsed
            self.face_time += elapsed  # Increment face time

            # self.ratio = self.no_face_time / self.face_time if self.face_time != 0 else 0
            self.avg_distance = sum(self.distance_history) / len(self.distance_history) if len(self.distance_history) != 0 else 0

            self.title = f"😊 {self.face_time:.1f}m | 🫥 {self.no_face_time:.1f}m | {self.avg_distance:.1f}cm🔺"

            if self.state_change_time >= self.face_time_threshold:
                face = faces[0]
                pointLeft = face[145]
                pointRight = face[374]

                w, _ = self.detector.findDistance(pointLeft, pointRight)
                W = 6.9  # Approximate width of face in cm
                f = 1700  # Approximate focal length
                d = (W * f) / w  # Calculated distance in cm

                # Store in rolling history
                self.distance_history.append(d)

                # Update menu bar title with distance and face timer
                if d > self.avg_distance:
                    if self.face_time > 40:
                        self.title = f"🫠 {self.face_time:.1f}m ({self.face_time_count})| 🫥 {self.no_face_time:.1f}m ({self.no_face_time_count})| {self.avg_distance:.1f}cm🔺"
                    else:
                        self.title = f"😊 {self.face_time:.1f}m ({self.face_time_count})| 🫥 {self.no_face_time:.1f}m ({self.no_face_time_count})| {self.avg_distance:.1f}cm🔺"
                else:
                    if self.face_time > 40:
                        self.title = f"🫠 {self.face_time:.1f}m ({self.face_time_count})| 🫥 {self.no_face_time:.1f}m ({self.no_face_time_count})| {self.avg_distance:.1f}cm🔻"
                    else:
                        self.title = f"😊 {self.face_time:.1f}m ({self.face_time_count})| 🫥 {self.no_face_time:.1f}m ({self.no_face_time_count})| {self.avg_distance:.1f}cm🔻"

                if not self.face_detected:  # If state changed from no face to face
                    self.state_change_time = 0
                    self.face_detected = True
                    # self.face_time = 0
                    self.no_face_time_count += 1
    
            if not self.face_detected:  # If state changed from no face to face
                self.face_detected = True
        
        else:
            self.state_change_time += elapsed
            self.no_face_time += elapsed  # Increment no face time

            # self.ratio = self.no_face_time / self.face_time if self.face_time != 0 else 0

            self.title = f"🫥 {self.no_face_time:.1f}m ({self.no_face_time_count})| 😐 {self.face_time:.1f}m ({self.face_time_count})"
            if self.state_change_time >= self.face_time_threshold:
                if self.face_detected:  # If state changed from face to no face
                    self.state_change_time = 0
                    self.face_detected = False
                    # self.no_face_time = 0
                    self.face_time_count += 1
            
            if self.face_detected:  # If state changed from face to no face
                self.face_detected = False
            
        print(f"State time:\t{self.state_change_time:.3f}")
        print(f"Face?\t\t{self.face_detected}")
        print(f"Face time:\t{self.face_time:.3f}")
        print(f"No face time:\t{self.no_face_time:.3f}")
        print()

        # # Check if the state has been in place for more than 30 seconds
        # if self.face_detected and self.face_time >= 30:
        #     self.title = "Face detected for over 30 seconds"
        # elif not self.face_detected and self.no_face_time >= 30:
        #     self.title = "No face detected for over 30 seconds"

if __name__ == "__main__":
    app = ScreenDistanceApp()
    app.run()
