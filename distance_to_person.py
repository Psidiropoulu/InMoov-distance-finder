import cv2
import numpy as np

# Load the MobileNet SSD model
net = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'ssd_mobilenet_v2_coco_2018_03_29.pbtxt')

# Define a simple color for the 'person' class, and maybe a default one for other classes
COLORS = [(0, 255, 0)]  # Green for 'person'

# Function to get the output layer names in the architecture
def get_output_layers(net):
    layer_names = net.getLayerNames()
    return [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def distance_to_camera(knownHeight, focalLength, perHeight):
    # Compute and return the distance from the marker to the camera
    return (knownHeight * focalLength) / perHeight

# Initialize the known parameter
KNOWN_HEIGHT = 170  # Average height of a person in cm
FOCAL_LENGTH = 700  # This needs to be pre-calculated

# Classes of objects MobileNet SSD was trained on
classes = ["background", "person", "bicycle", "car", "motorcycle", "airplane",
           "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
           "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
           "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
           "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
           "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
           "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
           "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
           "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
           "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor",
           "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
           "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
           "scissors", "teddy bear", "hair drier", "toothbrush"]

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Prepare the frame to be fed to the network
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    # Set the blob as input to the network and perform a forward pass to compute the detections
    net.setInput(blob)
    detections = net.forward(get_output_layers(net))

    # Loop over the detections
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        class_id = int(detections[0, 0, i, 1])
        
        if class_id == 1 and confidence > 0.5:
            # Scale the bounding box back to the size of the image
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Draw the prediction on the frame
            label = "{}: {:.2f}%".format(classes[class_id], confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Calculate the distance to the person
            personHeightInPixels = endY - startY
            distance = distance_to_camera(KNOWN_HEIGHT, FOCAL_LENGTH, personHeightInPixels)
            distance_label = f"Distance: {distance:.2f} cm"
            cv2.putText(frame, distance_label, (startX, endY + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Frame', frame)
    
    # Break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
