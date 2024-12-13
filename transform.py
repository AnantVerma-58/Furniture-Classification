import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
import cv2
from torchvision import transforms
def convert_images(input_dir, output_dir, target_format=".png"):
    """
    Convert all images in the input directory to a specific format and save them in the output directory,
    maintaining the original directory structure. Images are resized to 256x256 and converted to RGB.

    :param input_dir: Path to the input directory containing images.
    :param output_dir: Path to the output directory to save converted images.
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
                
                # Read and save the image
                input_file = os.path.join(root, file)
                image = cv2.imread(input_file)
                if image is not None:

                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
                    image_resized = cv2.resize(image_rgb, (256, 256))  # Resize to 256x256
                    
                    # Saving in new format
                    output_file = os.path.join(output_path, os.path.splitext(file)[0] + target_format)
                    cv2.imwrite(output_file, cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR))  # Convert back to BGR for OpenCV
                    # print(f"Converted and resized: {input_file} -> {output_file}")
                else:
                    print(f"Failed to read: {input_file}")


transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image (optional, based on model input)
    transforms.ToTensor(),  # Convert to Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])
