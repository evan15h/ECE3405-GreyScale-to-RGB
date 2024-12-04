import tensorflow as tf
import numpy as np
import os
import cv2
import tensorflow as tf

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# CIFAR-10 class labels
class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Filter only "dog" images (class index 5)
dog_class_index = 5
dog_images = x_train[y_train.flatten() == dog_class_index]
dog_labels = y_train[y_train.flatten() == dog_class_index]

# Save the dog images as grayscale
output_dir = "data/CIFAR10DogsGrayscale"
os.makedirs(output_dir, exist_ok=True)

for i, image in enumerate(dog_images):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Save image
    output_path = os.path.join(output_dir, f"dog_{i}.jpg")
    cv2.imwrite(output_path, gray_image)

    if i % 100 == 0:  # Print progress every 100 images
        print(f"Saved {i}/{len(dog_images)} grayscale images")

print("All dog images have been saved in grayscale format.")
