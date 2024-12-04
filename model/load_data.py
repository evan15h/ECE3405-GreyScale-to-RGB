import os
import numpy as np
import cv2
#pylint: disable=import-error
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def load_stanford_dogs_data(data_dir, image_size=(64, 64)):
    grayscale_images = []
    rgb_images = []
    labels = []
    breed_mapping = {breed: idx for idx, breed in enumerate(os.listdir(data_dir))}
    
    for breed, label in breed_mapping.items():
        breed_dir = os.path.join(data_dir, breed)
        
        for file_name in os.listdir(breed_dir):
            file_path = os.path.join(breed_dir, file_name)
            
            # Read RGB image
            rgb_image = cv2.imread(file_path)
            if rgb_image is None:
                print(f"Skipping invalid image: {file_path}")
                continue
            
            # Resize and normalize RGB image (convert to float32)
            rgb_image = cv2.resize(rgb_image, image_size).astype('float32') / 255.0
            rgb_images.append(rgb_image)
            
            # Convert to grayscale (cv2 expects uint8 input for color conversions)
            gray_image = cv2.cvtColor((rgb_image * 255).astype('uint8'), cv2.COLOR_RGB2GRAY)
            grayscale_images.append(np.expand_dims(gray_image, axis=-1))  # Add channel dimension
            
            # Append the label
            labels.append(label)
    
    # Convert lists to numpy arrays
    grayscale_images = np.array(grayscale_images)
    rgb_images = np.array(rgb_images)
    labels = to_categorical(np.array(labels), num_classes=len(breed_mapping))  # One-hot encode labels
    
    return grayscale_images, rgb_images, labels, breed_mapping


def split_data(X_gray, Y_rgb, y_labels, test_size=0.2):
    return train_test_split(
        X_gray, Y_rgb, y_labels, test_size=test_size, random_state=42
    )
