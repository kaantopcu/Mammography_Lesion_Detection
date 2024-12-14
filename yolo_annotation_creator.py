import os
from tqdm import tqdm

def create_yolo_annotations(df, base_yolo_label_dir, base_image_dir):
    """
    Creates YOLO annotation files for bounding boxes based on the provided dataframe.

    :param df: DataFrame containing annotation data with columns for finding categories and bounding box coordinates.
    :param base_yolo_label_dir: Base directory where YOLO label files will be saved.
    :param base_image_dir: Directory containing the images for the YOLO format.
    """
    # Define the mapping of finding categories to class IDs (0 to 6)
    category_to_id = {
        'No Finding': 0,
        'Mass': 1,
        'Suspicious Calcification': 2,
        'Architectural Distortion': 3,
        'Asymmetry': 4,
        'Focal Asymmetry': 5,
        'Global Asymmetry': 6
    }

    # Create split directories for labels
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(base_yolo_label_dir, split), exist_ok=True)

    # Counter dictionary for tracking label and image counts
    counts = {split: {'labels': 0, 'images': 0} for split in splits}

    # Iterate through the DataFrame to create YOLO `.txt` files and place them in the correct split
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # Extract relevant information
        unique_id = row['index']  # Unique ID for the bounding box
        finding_category = row['finding_categories']  # Finding category
        split = row['split']  # 'train', 'val', or 'test'

        # Define the output directory for the split
        split_label_dir = os.path.join(base_yolo_label_dir, split)
        label_file_path = os.path.join(split_label_dir, f"{unique_id}.txt")

        # Check if the finding is 'No Finding'
        if finding_category == 'No Finding':
            # Create an empty `.txt` file for "No Finding"
            open(label_file_path, 'w').close()
            counts[split]['labels'] += 1
        else:
            # Map the finding category to a class ID
            class_id = category_to_id.get(finding_category)
            if class_id is None:
                continue

            # Get bounding box coordinates and image dimensions
            xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
            img_width, img_height = row['width'], row['height']

            # Normalize bounding box coordinates for YOLO format
            x_center = (xmin + xmax) / 2 / img_width
            y_center = (ymin + ymax) / 2 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            # Create the YOLO label string
            label = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

            # Write the label to the file
            with open(label_file_path, 'w') as label_file:
                label_file.write(label)

            counts[split]['labels'] += 1

        # Check if the corresponding image exists in the image directory
        split_image_dir = os.path.join(base_image_dir, split)
        image_file_path = os.path.join(split_image_dir, f"{unique_id}.png")
        if os.path.exists(image_file_path):
            counts[split]['images'] += 1

    # Print summary of counts
    print("Summary of Images and Labels per Split:")
    for split in splits:
        print(f"Split: {split}")
        print(f"  Number of label files: {counts[split]['labels']}")
        print(f"  Number of images: {counts[split]['images']}")
        print()