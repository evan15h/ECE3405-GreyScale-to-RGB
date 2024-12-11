import os
import numpy as np
import cv2
#pylint: disable=import-error
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split



def load_stanford_dogs_data(data_dir, mask_dir, image_size=(128, 128)):
    grayscale_images = []
    rgb_images = []
    labels = []
    mask_images = []  
    all_breeds = os.listdir(data_dir)
    mask_breeds = os.listdir(mask_dir)
    print("Folders found in data_dir:", all_breeds)
    print("Folders found in mask_dir:", mask_breeds)  

    # Filter folders if selected_breeds is provided
    # if selected_breeds:
    #     all_breeds = [breed for breed in all_breeds if breed in selected_breeds]

    breed_mapping = {breed: idx for idx, breed in enumerate(all_breeds)}
    mask_mapping = {mask: idx for idx, mask in enumerate(mask_breeds)}
    print(f"Filtered breeds: {all_breeds}")
    print(f"Breed mapping: {breed_mapping}")
    print(f"Mask mapping: {mask_mapping}")

    
    for breed, label in breed_mapping.items():
        breed_dir = os.path.join(data_dir, breed)
        mask_breed_dir = os.path.join(mask_dir, f"Masked {breed}")
        
        for file_name in os.listdir(breed_dir):
            file_path = os.path.join(breed_dir, file_name)
            mask_path = os.path.join(mask_breed_dir, f"mask_for_{file_name}")
            
            # Read RGB image
            image = cv2.imread(file_path)
            if image is None:
                print(f"Skipping invalid image: {file_path}")
                continue

            mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_image is None:
                print(f"Skipping missing mask: {mask_path}")
                continue
            
            # Resize and normalize RGB image (convert to float32)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            rgb_image = cv2.resize(rgb_image, image_size).astype('float32') / 255.0
            
            # Convert to grayscale (cv2 expects uint8 input for color conversions)
            gray_image = cv2.cvtColor((rgb_image * 255).astype('uint8'), cv2.COLOR_RGB2GRAY)

            combined_input = np.stack((gray_image, mask_image), axis=-1)  # (H, W, 2)

            grayscale_images.append(combined_input)
            rgb_images.append(rgb_image)
            mask_images.append(mask_image)
            labels.append(label)
    
    # Debugging prints
    print(f"Number of grayscale images: {len(grayscale_images)}")
    print(f"Number of RGB images: {len(rgb_images)}")
    print(f"Number of masks: {len(mask_images)}")
    print(f"Number of labels: {len(labels)}")

    if not grayscale_images or not rgb_images or not labels:
        raise ValueError("No images or labels were loaded. Check the dataset directory or selected breeds.")

    # Convert lists to numpy arrays
    grayscale_images = np.array(grayscale_images)
    rgb_images = np.array(rgb_images)
    labels = to_categorical(np.array(labels), num_classes=len(breed_mapping))  # One-hot encode labels
    
    return grayscale_images, rgb_images, labels, breed_mapping


def split_data(X_gray, Y_rgb, y_labels, test_size=0.2):
    return train_test_split(
        X_gray, Y_rgb, y_labels, test_size=test_size, random_state=42
    )
