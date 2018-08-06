# Webcam face detector using LBP Cascades
# TUTORIAL AND CODE: https://www.superdatascience.com/opencv-face-detection/

import cv2
import numpy as np

cap = cv2.VideoCapture(0)	                        # number in argument is video source, my webcam is 0
face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface_improved.xml')     # LBP face detector

faces = None
while True:			                                # capture frame-by-frame
    ret, frame = cap.read()                         # grab the current frame
    frame = cv2.flip(frame, 1)	                    # horizontal flip on the frame, so it acts as a mirror
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

    cv2.imshow('HAAR', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):                                           # Press Q to quit
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
