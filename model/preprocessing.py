import tensorflow as tf
import numpy as np
import cv2
import os
import tensorflow_hub as hub
import torch

def detect_and_crop_with_yolo(input_dir, output_dir, model_path='yolov5s', target_size=(64, 64)):
    """
    Detects dogs in images using YOLOv5 and crops around them.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load YOLOv5 model (pretrained weights)
    model = torch.hub.load('./yolov5', 'yolov5s', source='local')  # Load YOLOv5 locally

    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)
        if not filename.lower().endswith(('png', 'jpg', 'jpeg')):
            continue

        # Load image
        image = cv2.imread(filepath)
        if image is None:
            continue

        # Run YOLO detection
        results = model(filepath)
        detections = results.pandas().xyxy[0]  # Get bounding boxes as a pandas DataFrame

        # Filter detections for 'dog' class (YOLO class index may vary)
        dog_detections = detections[detections['name'] == 'dog']
        if dog_detections.empty:
            continue

        # Use the largest bounding box
        largest_box = dog_detections.iloc[0]
        x1, y1, x2, y2 = int(largest_box['xmin']), int(largest_box['ymin']), int(largest_box['xmax']), int(largest_box['ymax'])

        # Crop the image
        cropped = image[y1:y2, x1:x2]

        # Resize to target size
        resized = cv2.resize(cropped, target_size)

        # Save the processed image
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, resized)

input_dir = "data/Images/n02113799-standard_poodle"
output_dir = "data/Preprocessed/Standard Poodle"
detect_and_crop_with_yolo(input_dir, output_dir)