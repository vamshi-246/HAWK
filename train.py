from roboflow import Roboflow
from ultralytics import YOLO
import os

# --- 1. DOWNLOAD DATASET ---
# if not os.path.exists("cows-roboflow-dataset"):
#     print("Downloading dataset...")
#     rf = Roboflow(api_key="API_KEY")
#     project = rf.workspace("workspace").project("hawk")
#     version = project.version(1)
#     dataset = version.download("yolov8")
# else:
#     print("Dataset already exists.")

# --- 2. TRAIN THE MODEL ---

model = YOLO('yolov8n.pt')

# Train the model on your custom dataset
results = model.train(
    data='HAWK-WINDOWS-1/data.yaml',
    epochs=5,  
    imgsz=640,  
    batch=4,    
    name='yolov8n_cows_v1' 
)

print("Training complete! Model saved in the 'runs' folder.")

