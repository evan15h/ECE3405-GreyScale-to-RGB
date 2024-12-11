import numpy as np
import cv2
import os
import torch
import torchvision
from torchvision.transforms import functional as F

def detect_and_crop_with_yolo(input_dir, output_dir, model_path='yolov5s', target_size=(128, 128)):
    """
    Detects dogs in images using YOLOv5 and crops around them.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load YOLOv5 model (pretrained weights)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

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
        save_path = os.path.join(output_dir, f"cropped_{filename}")
        cv2.imwrite(save_path, resized)

def save_segmentation_masks_maskrcnn(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Load pre-trained Mask R-CNN model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()  # Set to evaluation mode

    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)
        if not filename.lower().endswith(('png', 'jpg', 'jpeg')):
            continue

        image = cv2.imread(filepath)
        if image is None:
            print(f"Skipping invalid or unreadable file: {filepath}")
            continue

        # Convert to PIL-compatible format and normalize
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = F.to_tensor(image_rgb).unsqueeze(0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            outputs = model(image_tensor)

        # Extract masks and scores
        masks = outputs[0]['masks']  # Shape: [N, 1, H, W]
        scores = outputs[0]['scores']  # Confidence scores
        labels = outputs[0]['labels']  # Predicted classes

        # Filter masks by score threshold
        threshold = 0.5
        selected_masks = masks[scores > threshold].squeeze(1).cpu().numpy()

        if selected_masks.size == 0:
            print(f"No masks generated for {filename}")
            continue

        # Combine all masks into one binary mask
        combined_mask = np.zeros_like(image_rgb[:, :, 0], dtype=np.uint8)
        for mask in selected_masks:
            binary_mask = (mask > 0.5).astype(np.uint8) * 255
            combined_mask = np.maximum(combined_mask, binary_mask)

        # Save the combined mask
        save_path = os.path.join(output_dir, f"mask_for_{filename}")
        cv2.imwrite(save_path, combined_mask)
        print(f"Mask saved: {save_path}")

input_dir = "/Users/paigerust/Desktop/ECE3405/Final/Images/n02091134-whippet"
output_dir = "/Users/paigerust/Desktop/ECE3405/Final/Preprocessed/Whippet"
masked_output_dir = "/Users/paigerust/Desktop/ECE3405/Final/Masks/Whippet"
detect_and_crop_with_yolo(input_dir, output_dir)
save_segmentation_masks_maskrcnn(output_dir, masked_output_dir) #Create segmentation masks for cropped images