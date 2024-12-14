import os
import json
from tqdm import tqdm

# Define the mapping of finding categories to class IDs (1-based for COCO)
category_to_id = {
    'No Finding': 0,  # Special case, not included in "categories"
    'Mass': 1,
    'Suspicious Calcification': 2,
    'Architectural Distortion': 3,
    'Asymmetry': 4,
    'Focal Asymmetry': 5,
    'Global Asymmetry': 6
}

# Initialize COCO annotation structure
coco_annotations = {
    "images": [],
    "annotations": [],
    "categories": [
        {"id": class_id, "name": category}
        for category, class_id in category_to_id.items() if class_id != 0  # Exclude 'No Finding'
    ]
}

# Annotation ID counter
annotation_id = 1

# Base image directory
base_image_dir = "/content/drive/MyDrive/İlkay Hoca/final_yolo/images"

# Iterate through the DataFrame to populate the COCO structure
for _, row in tqdm(df.iterrows(), total=len(df)):
    # Extract relevant information
    unique_id = row['index']  # Unique ID for the image
    finding_category = row['finding_categories']  # Finding category
    split = row['split']  # 'training', 'validation', or 'test'

    # Get image dimensions
    img_width, img_height = row['width'], row['height']
    image_info = {
        "id": unique_id,
        "file_name": f"{unique_id}.png",
        "width": img_width,
        "height": img_height
    }
    coco_annotations["images"].append(image_info)

    # Skip adding annotations for "No Finding"
    if finding_category == 'No Finding':
        continue

    # Map the finding category to a class ID
    class_id = category_to_id.get(finding_category)
    if class_id is None:
        continue

    # Get bounding box coordinates
    xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
    bbox_width = xmax - xmin
    bbox_height = ymax - ymin

    # Add the annotation
    annotation = {
        "id": annotation_id,
        "image_id": unique_id,
        "category_id": class_id,
        "bbox": [xmin, ymin, bbox_width, bbox_height],  # COCO uses [x_min, y_min, width, height]
        "area": bbox_width * bbox_height,
        "iscrowd": 0  # Set to 0 for regular annotations
    }
    coco_annotations["annotations"].append(annotation)
    annotation_id += 1

# Save the COCO annotations as a JSON file
for split in ['train', 'val', 'test']:
    split_annotations = {
        "images": [img for img in coco_annotations["images"] if img["file_name"].startswith(split)],
        "annotations": [ann for ann in coco_annotations["annotations"] if str(ann["image_id"]).startswith(split)],
        "categories": coco_annotations["categories"]
    }
    output_path = os.path.join(base_image_dir, f"{split}_annotations.json")
    with open(output_path, 'w') as json_file:
        json.dump(split_annotations, json_file, indent=4)

print("COCO annotation files created for each split!")
