# yolomodel/detector.py
import cv2
import numpy as np
from ultralytics import YOLO

class TrafficDetector:
    def __init__(self, model_path, mask_points):
        self.model = YOLO(model_path)
        self.mask_points = mask_points

    def draw_polygon(self, img, points):
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        if points:
            cv2.fillPoly(mask, [np.array(points)], 255)
        return mask

    def detect_vehicles(self, img):
        mask = self.draw_polygon(img, self.mask_points)
        results = self.model(img)
        vehicle_counts = {'car': 0, 'motorbike': 0, 'bus': 0, 'truck': 0}
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                if mask[center_y, center_x] == 255:
                    name = result.names[int(box.cls[0])]
                    if name in vehicle_counts:
                        vehicle_counts[name] += 1
        return vehicle_counts
