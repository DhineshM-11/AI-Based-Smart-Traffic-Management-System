# 🚦 AI Smart Traffic Management System

A traffic analysis app for vehicle detection, traffic density evaluation, and emergency vehicle prioritization.

## Project Overview

This project uses a Streamlit web interface to upload traffic images from up to four roads, analyzes vehicle detections via Roboflow inference, and provides traffic density and signal priority suggestions. It highlights emergency vehicles such as ambulances, fire trucks, and police vehicles.

## Project 

- `app.py` - Main Streamlit application.
- `detection.py` - Local YOLOv8-based vehicle detection helper (note: current app code uses Roboflow inference in `app.py`).
- `yolov8n.pt` - YOLOv8 model weights used by the local detection script.

## Requirements

- Python 3.8+
- `streamlit`
- `inference_sdk`
- `Pillow`
- `ultralytics`
- `opencv-python`
- `numpy`

## Install Dependencies

```bash
pip install streamlit Pillow ultralytics opencv-python numpy inference_sdk
```
## Run the App

From the project root directory, start the Streamlit app:

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

## Usage

1. Upload up to four road images using the uploader panels.
2. The app sends each image through the Roboflow inference workflow.
3. Detected vehicles are drawn on the image with labels.
4. The app calculates traffic density and highlights emergency vehicles.
5. If an emergency vehicle is detected, the app recommends signal priority based on vehicle type and distance.

## Notes

- `app.py` currently relies on a Roboflow inference workflow and API key defined inside the file.
- For an offline detection workflow, `detection.py` can be used with the local `yolov8n.pt` model.
- Keep the `yolov8n.pt` file in the project root if you plan to use local detection.
