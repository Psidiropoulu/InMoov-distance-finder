# import the necessary packages
from imutils import paths
import numpy as np
import imutils
import cv2

def find_marker(image):
    # Convert the image to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect edges in the image
    edged = cv2.Canny(gray, 35, 125)
    
    # Find contours in the edged image and keep the largest one
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key = cv2.contourArea)
        return cv2.minAreaRect(c)
    return None

def distance_to_camera(knownWidth, focalLength, perWidth):
    # Compute and return the distance from the maker to the camera
    return (knownWidth * focalLength) / perWidth

# Initialize the known parameters
KNOWN_WIDTH = 10.0 # Width of the object in cm (change this to your object's width)
FOCAL_LENGTH = 700 # This needs to be pre-calculated

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Find the marker in the frame
    marker = find_marker(frame)
    if marker:
        perWidth = marker[1][0]
        # Calculate the distance to the marker
        distance = distance_to_camera(KNOWN_WIDTH, FOCAL_LENGTH, perWidth)
        print(f"Distance to object: {distance} cm")

        # You can also draw a bounding box around the object and display the distance
        box = cv2.boxPoints(marker)
        box = np.int0(box)
        cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)
        cv2.putText(frame, f"{distance:.2f} cm", (frame.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Frame', frame)
    
    # Break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
