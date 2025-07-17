from ultralytics import YOLO
import cv2
path = 'runs/detect/yolov8n_cows_v12/weights/best.pt'

# Other options: 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'
model = YOLO(path)


source = 0 # Replace with your image path or 0 for webcam

# Perform inference
# The 'stream=True' argument is more memory-efficient for videos or webcam
results = model(source, show=True, conf=0.4, save=True)



# for r in results:
#     boxes = r.boxes  # Bounding box objects
#     for box in boxes:
#         print(f"Detected: {model.names[int(box.cls)]} with confidence {box.conf.item():.2f}")

print("Detection complete. Press any key in the display window to exit.")
cv2.waitKey(0) 
