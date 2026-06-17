import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def detect_vehicles(image):
    results = model(image, conf=0.25)

    vehicle_counts = {"car":0,"motorcycle":0,"bus":0,"truck":0}
    emergency = False

    vehicle_detected = False  # 🔥 Track if any vehicle present

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label in vehicle_counts:
                vehicle_counts[label] += 1
                vehicle_detected = True

    # -------------------- COLOR DETECTION --------------------
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Red mask
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170,120,70])
    upper_red2 = np.array([180,255,255])

    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) + \
               cv2.inRange(hsv, lower_red2, upper_red2)

    # Blue mask
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    red_pixels = np.sum(red_mask > 0)
    blue_pixels = np.sum(blue_mask > 0)

    # -------------------- SMART CONDITION --------------------
    if vehicle_detected and (red_pixels > 800 or blue_pixels > 800):
        emergency = True

    detected_img = results[0].plot()

    return detected_img, vehicle_counts, emergency