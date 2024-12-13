import os
import cv2
import numpy as np

def augment_images(input_dir, output_dir, target_size=(256, 256), target_format=".png"):
    """
    Augment all images in the input directory and save 5 augmented versions per image
    to the output directory, maintaining the original directory structure.

    :param input_dir: Path to the input directory containing images.
    :param output_dir: Path to the output directory to save augmented images.
    :param target_size: Tuple indicating the size (width, height) to resize images.
    :param target_format: Target image format (e.g., ".png", ".jpg").
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Output directory created: {output_dir}")
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            # Check if file is an image
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                # Create the corresponding output directory
                relative_path = os.path.relpath(root, input_dir)
                output_path = os.path.join(output_dir, relative_path)
                os.makedirs(output_path, exist_ok=True)

                # Read the image
                input_file = os.path.join(root, file)
                image = cv2.imread(input_file)
                if image is not None:
                    # Resize and convert to RGB
                    image_resized = cv2.resize(image, target_size)
                    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
                    
                    # Generate augmentations
                    augmentations = [
                        image_rgb,  # Original resized image
                        cv2.flip(image_rgb, 1),  # Horizontal flip
                        cv2.rotate(image_rgb, cv2.ROTATE_90_CLOCKWISE),  # Rotate 90 degrees
                        apply_blur(image_rgb),
                        adjust_brightness(image_rgb, factor=1.2),  # Brightness increase
                        add_noise(image_rgb)  # Add random noise
                    ]
                    
                    # Save augmented images
                    for i, aug_image in enumerate(augmentations):
                        output_file = os.path.join(
                            output_path,
                            f"{os.path.splitext(file)[0]}_aug{i+1}{target_format}"
                        )
                        cv2.imwrite(output_file, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
                        # print(f"Augmented and saved: {output_file}")
                else:
                    print(f"Failed to read image: {input_file}")

def apply_blur(image, kernel_size=(5, 5)):
    """
    Apply Gaussian blur to an image.
    :param image: Input image (RGB).
    :param kernel_size: Size of the Gaussian kernel (odd integers).
    :return: Blurred image.
    """
    return cv2.GaussianBlur(image, kernel_size, 0)


def adjust_brightness(image, factor=1.0):
    """
    Adjust the brightness of an image.
    :param image: Input image (RGB).
    :param factor: Brightness factor (e.g., 1.2 for 20% increase).
    :return: Brightness-adjusted image.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def add_noise(image):
    """
    Add random Gaussian noise to an image.
    :param image: Input image (RGB).
    :return: Image with added noise.
    """
    noise = np.random.normal(0, 25, image.shape).astype(np.float32)
    noisy_image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy_image

#Example
#augment_images(input_directory, output_directory, target_size=(256, 256), target_format=".png")
