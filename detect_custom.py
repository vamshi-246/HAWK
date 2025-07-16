from ultralytics import YOLO
import cv2
path = 'runs/detect/yolov8n_cows_v12/weights/best.pt'
# Load a pre-trained YOLOv8 model
# 'yolov8n.pt' is the smallest and fastest model.
# Other options: 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'
model = YOLO(path)

# Define the image source
# Can be a path to an image file or 0 for the webcam
source = 0 # Replace with your image path or 0 for webcam

# Perform inference
# The 'stream=True' argument is more memory-efficient for videos or webcam
results = model(source, show=True, conf=0.4, save=True)

# The 'show=True' argument automatically displays the results.
# The 'save=True' argument automatically saves the results in a 'runs/detect/' folder.
# 'conf=0.4' means it will only show detections with a confidence score of 40% or higher.

# The code above is sufficient for display. If you want to manually process results
# (e.g., for a webcam stream), you would loop through them like this:
#
# for r in results:
#     boxes = r.boxes  # Bounding box objects
#     for box in boxes:
#         # ... you can access box coordinates, class, and confidence here ...
#         print(f"Detected: {model.names[int(box.cls)]} with confidence {box.conf.item():.2f}")

print("Detection complete. Press any key in the display window to exit.")
cv2.waitKey(0) # Keep window open until a key is pressed (if using an image)
