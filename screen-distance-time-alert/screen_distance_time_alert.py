import cv2
from cvzone.FaceMeshModule import FaceMeshDetector
import time
import collections

cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)

time_detected = 0
time_not_detected = 0
last_state = None  # None, 'detected', or 'not_detected'
start_time = time.time()
last_print_time = 0  # Track last print timestamp

distance_history = collections.deque(maxlen=100)  # Store last 100 measurements

while True:
    success, img = cap.read()
    img, faces = detector.findFaceMesh(img, draw=False)

    # Sleep to slow measurement to save resources/power
    time.sleep(0.5)

    current_state = 'detected' if faces else 'not_detected'
    if faces:
        face = faces[0]
        pointLeft = face[145]
        pointRight = face[374]

        w, _ = detector.findDistance(pointLeft, pointRight)
        W = 6.9  # Approximate width of face between points in cm

        # Find distance
        f = 1700  # Approximate focal length
        d = (W * f) / w
        distance_history.append(d)  # Store distance in history

        # Compute rolling mean
        rolling_mean = sum(distance_history) / len(distance_history)

        print(f"Distance: {d:.2f} cm (Rolling Mean: {rolling_mean:.2f} cm)")

        # Beep if too close
        limit = 50
        if d < limit:
            print('\a')

    if current_state != last_state:
        # Reset timer if state changes
        start_time = time.time()
    
    elapsed_time = time.time() - start_time
    elapsed_minutes = elapsed_time / 60

    current_time = time.time()
    if current_time - last_print_time >= 30:  # Print only if 30 seconds have passed
        last_print_time = current_time
        print(f"Elapsed time: {elapsed_minutes:.1f} minutes")

    last_state = current_state