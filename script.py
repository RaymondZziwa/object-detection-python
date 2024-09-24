import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet("model/yolov3.weights", "model/yolov3.cfg")  # Ensure the paths are correct
classes = []
with open("model/coco.names", "r") as f:  # Path to the coco.names file
    classes = [line.strip() for line in f.readlines()]

# Get the names of all the layers in the network
layer_names = net.getLayerNames()

output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load an image
img = cv2.imread("dataset/data2.jpg")  # Ensure the image path is correct
#img = cv2.resize(img, None, fx=0.4, fy=0.4)  # Resize for faster processing
height, width, channels = img.shape

# Create a blob from the image
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Extract information from the output
class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]  # The confidence scores for each class
        class_id = np.argmax(scores)  # Find the class with the highest score
        confidence = scores[class_id]  # Get the confidence for the detected class
        if confidence > 0.5:  # Confidence threshold
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Coordinates for the bounding box
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Non-Maximum Suppression to remove overlapping boxes
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Display the result
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = (0, 255, 0)  # Green color for bounding box
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, f"{label} {round(confidence, 2)}", (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

# Show the image with detections
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
