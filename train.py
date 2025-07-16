from roboflow import Roboflow
from ultralytics import YOLO
import os

# --- 1. DOWNLOAD DATASET ---
# NOTE: You only need to run this once!
# After the first run, you can comment out this section.

# Check if the dataset already exists
if not os.path.exists("cows-roboflow-dataset"):
    print("Downloading dataset...")
    rf = Roboflow(api_key="tE7LqEnCvqAvJfnF0r74")
    project = rf.workspace("vamshi-9njef").project("hawk-windows")
    version = project.version(1)
    dataset = version.download("yolov8")
else:
    print("Dataset already exists.")

# --- 2. TRAIN THE MODEL ---

# Load a pre-trained model as a starting point.
# 'yolov8n.pt' is the nano version, smallest and fastest.
# 'yolov8s.pt' (small) is another good starting point.
model = YOLO('yolov8n.pt')

# Train the model on your custom dataset
# The 'data' argument should point to the 'data.yaml' file in your downloaded dataset.
results = model.train(
    data='HAWK-WINDOWS-1/data.yaml',
    epochs=5,  # Number of times to train on the entire dataset. Start with 50-100.
    imgsz=640,  # Image size to train on. 640 is a standard.
    batch=4,    # Number of images per batch. Lower if you get memory errors.
    name='yolov8n_cows_v1' # A custom name for this training run
)

print("Training complete! Model saved in the 'runs' folder.")

