# HAWK
# Animal Object Detection (COWS)

This project is an object detection system built with Python, OpenCV, and YOLOv8 to detect cows in images and video streams.

## Features

- Real-time detection using a webcam.
- Detection on static images.
- Trained on a custom dataset from Roboflow.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/vamshi-246/HAWK.git
    cd HAWK
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run a prediction:**
    ```bash
    python detect_custom.py
    ```

## Trained Model

The trained model `best.pt` is not included in this repository due to its size. You can download it from the [Releases page](https://github.com/vamshi-246/HAWK/releases). 

Place the downloaded `best.pt` file in the correct path as specified in the detection script.
