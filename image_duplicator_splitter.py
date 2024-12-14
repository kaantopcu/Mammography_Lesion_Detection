import os
import shutil
import pandas as pd

def duplicate_and_split_images(df, base_image_dir, output_image_dir, base_split_dir):
    """
    Duplicates images based on annotations and splits them into train, validation, and test directories.
    
    :param df: DataFrame containing the annotations, including 'study_id', 'image_id', and 'split'.
    :param base_image_dir: Directory where the original images are stored.
    :param output_image_dir: Directory to store duplicated images temporarily.
    :param base_split_dir: Base directory where the train/val/test split directories are located.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_image_dir, exist_ok=True)

    # Create split directories if they don't exist
    splits = ['training', 'validation', 'test']
    for split in splits:
        os.makedirs(os.path.join(base_split_dir, 'images', split), exist_ok=True)

    # Iterate through annotations to duplicate images and move them to respective split directories
    for _, row in df.iterrows():
        study_id = row['study_id']
        image_id = row['image_id']
        unique_id = row['index']  # Unique identifier for the bounding box
        split = row['split']  # 'train', 'val', or 'test'

        # Source and destination paths
        image_src = os.path.join(base_image_dir, study_id, f"{image_id}.png")
        image_dst = os.path.join(output_image_dir, f"{unique_id}.png")

        # Copy image if it exists (Duplicate image based on bounding box)
        if os.path.exists(image_src):
            shutil.copy(image_src, image_dst)

            # Move the image to the appropriate split folder
            image_split_dst = os.path.join(base_split_dir, 'images', split, f"{unique_id}.png")
            shutil.move(image_dst, image_split_dst)
            print(f"Moved {image_id}.png to {split} split.")