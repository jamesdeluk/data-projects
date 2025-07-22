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
        
        # Time tracking for different states
        self.face_time = 0  # Time with face detected
        self.no_face_time = 0  # Time with no face detected
        self.last_time = time.time()
        self.face_detected = False  # Tracks the last state

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
            if not self.face_detected:  # If state changed from no face to face
                # self.face_time = 0
                self.face_detected = True
            
            self.face_time += elapsed  # Increment face time
            face = faces[0]
            pointLeft = face[145]
            pointRight = face[374]

            w, _ = self.detector.findDistance(pointLeft, pointRight)
            W = 6.9  # Approximate width of face in cm
            f = 1700  # Approximate focal length
            d = (W * f) / w  # Calculated distance in cm

            # Store in rolling history
            self.distance_history.append(d)
            avg_distance = sum(self.distance_history) / len(self.distance_history)

            ratio = self.no_face_time / self.face_time

            # Update menu bar title with distance and face timer
            if d > avg_distance:
                self.title = f"😊 {self.face_time:.1f}m | {int(100*ratio)}% | {avg_distance:.1f}cm🔺"
            else:
                self.title = f"😊 {self.face_time:.1f}m | {int(100*ratio)}% | {avg_distance:.1f}cm🔻"
        
        else:
            if self.face_detected:  # If state changed from face to no face
                # self.no_face_time = 0
                self.face_detected = False

            self.no_face_time += elapsed  # Increment no face time
            ratio = self.no_face_time / self.face_time
            self.title = f"🫥 {self.no_face_time:.1f}m | {int(100*ratio)}%"

        # # Check if the state has been in place for more than 30 seconds
        # if self.face_detected and self.face_time >= 30:
        #     self.title = "Face detected for over 30 seconds"
        # elif not self.face_detected and self.no_face_time >= 30:
        #     self.title = "No face detected for over 30 seconds"

if __name__ == "__main__":
    app = ScreenDistanceApp()
    app.run()
