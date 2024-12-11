import os
import numpy as np
import cv2
#pylint: disable=import-error
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from torchvision import transforms
from PIL import Image

def load_stanford_dogs_data(data_dir, image_size=(64, 64), selected_breeds=None):
    grayscale_images = []
    rgb_images = []
    labels = []
    all_breeds = os.listdir(data_dir)
    print("Folders found in data_dir:", all_breeds) 
    
    print("Preparing data transformations...")
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.449, std=0.226)])

    # Filter folders if selected_breeds is provided
    if selected_breeds:
        all_breeds = [breed for breed in all_breeds if breed in selected_breeds]

    breed_mapping = {breed: idx for idx, breed in enumerate(all_breeds)}
    print(f"Filtered breeds: {all_breeds}")
    print(f"Breed mapping: {breed_mapping}")
    
    for breed, label in breed_mapping.items():
        breed_dir = os.path.join(data_dir, breed)

        if os.path.isdir(breed_dir):
            for file_name in os.listdir(breed_dir):
                file_path = os.path.join(breed_dir, file_name)
                
                # Read RGB image
                rgb_image = cv2.imread(file_path)
                if rgb_image is None:
                    print(f"Skipping invalid image: {file_path}")
                    continue
                
                # Resize and normalize RGB image (convert to float32)
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
                rgb_image = cv2.resize(rgb_image, image_size).astype('float32') / 255.0
                rgb_images.append(rgb_image)
                
                # Convert to grayscale (cv2 expects uint8 input for color conversions)
                gray_image = cv2.cvtColor((rgb_image * 255).astype('uint8'), cv2.COLOR_RGB2GRAY)
                grayscale_images.append(np.expand_dims(gray_image, axis=-1))  # Add channel dimension
                
                # Append the label
                labels.append(label)
    
    # Debugging prints
    print(f"Number of grayscale images: {len(grayscale_images)}")
    print(f"Number of RGB images: {len(rgb_images)}")
    print(f"Number of labels: {len(labels)}")

    if not grayscale_images or not rgb_images or not labels:
        raise ValueError("No images or labels were loaded. Check the dataset directory or selected breeds.")

    # Convert lists to numpy arrays
    grayscale_images = np.array(grayscale_images)
    rgb_images = np.array(rgb_images)
    labels = to_categorical(np.array(labels), num_classes=len(breed_mapping))  # One-hot encode labels

    grayscale_image_transform = np.empty([1, 64, 64, 1])

    for image in grayscale_images:
        image = image.reshape(64, 64)

        image = Image.fromarray(image, mode='L')

        image = transform(image)
        
        image = np.array(image)

        image = image.reshape(1, 64, 64, 1)

        grayscale_image_transform = np.vstack([grayscale_image_transform, image])

    grayscale_image_transform = np.delete(grayscale_image_transform, 0, 0)
    grayscale_images = grayscale_image_transform
        
    return grayscale_images, rgb_images, labels, breed_mapping


def split_data(X_gray, Y_rgb, y_labels, test_size=0.2):
    return train_test_split(
        X_gray, Y_rgb, y_labels, test_size=test_size, random_state=42
    )
