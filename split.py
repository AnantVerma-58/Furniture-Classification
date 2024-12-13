import os
import shutil
import random
from utils import augment_folder, final_data_folder
def split_dataset(input_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split dataset into train, validation, and test sets and save them in separate directories.
    
    :param input_dir: Path to the input directory containing class folders.
    :param output_dir: Path to the output directory to store train, val, and test folders.
    :param train_ratio: Proportion of data to use for training.
    :param val_ratio: Proportion of data to use for validation.
    :param test_ratio: Proportion of data to use for testing.
    """
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
    
    # Loop through each class folder in the input directory
    for class_folder in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_folder)
        
        if os.path.isdir(class_path):
            # Create subdirectories in the output directory for each class
            os.makedirs(os.path.join(output_dir, 'train', class_folder), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'val', class_folder), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'test', class_folder), exist_ok=True)

            # Get all image files in the class folder
            images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
            
            # Shuffle images to randomize selection
            random.shuffle(images)

            # Calculate number of samples for train, val, and test
            total_images = len(images)
            train_size = int(train_ratio * total_images)
            val_size = int(val_ratio * total_images)
            test_size = total_images - train_size - val_size

            # Split the images into train, validation, and test
            train_images = images[:train_size]
            val_images = images[train_size:train_size + val_size]
            test_images = images[train_size + val_size:]

            # Move images to respective folders
            for img in train_images:
                shutil.copy(os.path.join(class_path, img), os.path.join(output_dir, 'train', class_folder, img))
            for img in val_images:
                shutil.copy(os.path.join(class_path, img), os.path.join(output_dir, 'val', class_folder, img))
            for img in test_images:
                shutil.copy(os.path.join(class_path, img), os.path.join(output_dir, 'test', class_folder, img))

            print(f"Processed class: {class_folder}, {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")

# Example usage
# split_dataset(augment_folder, final_data_folder)