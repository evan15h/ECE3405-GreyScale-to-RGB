from PIL import Image
import cv2
import os

def convert_to_greyscale(input_path, output_path):
    """
    Converts an RGB image to greyscale and saves it.

    :param input_path: Path to the input RGB image.
    :param output_path: Path where the greyscale image will be saved.
    """
    try:
        # Open the input image
        img = Image.open(input_path)
        
        # Convert the image to greyscale
        greyscale_img = img.convert("L")
        
        # Save the greyscale image
        greyscale_img.save(output_path)
        print(f"Greyscale image saved to: {output_path}")
    except Exception as e:
        print(f"Error: {e}")

# Example usage:
if __name__ == "__main__":
    # Paths to the input RGB image and the output greyscale image
    input_image_path = "input_image.jpg"  # Change this to your input image path
    output_image_path = "output_image_greyscale.jpg"  # Change this to your desired output path
    
    # Convert the image
    convert_to_greyscale(input_image_path, output_image_path)
