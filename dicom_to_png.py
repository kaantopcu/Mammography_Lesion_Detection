import os
import pydicom
from PIL import Image

def get_dicom_files(input_dir):
    """
    Traverse the input directory and yield paths of all DICOM files.

    :param input_dir: Path to the root folder containing DICOM files.
    :return: Yields DICOM file paths.
    """
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.dcm', '.dicom')):
                yield os.path.join(root, file)

def convert_dicom_to_image(dicom_path):
    """
    Convert a DICOM file to a PNG image.

    :param dicom_path: Path to the DICOM file.
    :return: PIL Image object of the converted DICOM image.
    """
    try:
        dicom = pydicom.dcmread(dicom_path)
        pixel_array = dicom.pixel_array
        pixel_array = ((pixel_array - pixel_array.min()) /
                       (pixel_array.max() - pixel_array.min()) * 255).astype('uint8')
        return Image.fromarray(pixel_array)
    except Exception as e:
        print(f"Failed to convert {dicom_path}: {e}")
        return None

def save_image(image, output_path):
    """
    Save the PIL Image to the specified output path.

    :param image: PIL Image object.
    :param output_path: Path to save the PNG image.
    """
    try:
        image.save(output_path)
        print(f"Saved: {output_path}")
    except Exception as e:
        print(f"Failed to save image: {output_path} - {e}")

def create_output_directory(output_dir, dicom_path, input_dir):
    """
    Create the output directory while maintaining the folder structure.

    :param output_dir: The root folder to save PNG files.
    :param dicom_path: Path to the DICOM file.
    :param input_dir: Path to the root folder containing DICOM files.
    :return: The output directory path.
    """
    relative_path = os.path.relpath(os.path.dirname(dicom_path), input_dir)
    output_folder = os.path.join(output_dir, relative_path)
    os.makedirs(output_folder, exist_ok=True)
    return output_folder

def convert_dicom_files(input_dir, output_dir):
    """
    Convert all DICOM files in the input directory to PNG images and save them to the output directory.

    :param input_dir: Path to the root folder containing DICOM files.
    :param output_dir: Path to the root folder to save PNG files.
    """
    for dicom_path in get_dicom_files(input_dir):
        image = convert_dicom_to_image(dicom_path)
        if image:
            output_folder = create_output_directory(output_dir, dicom_path, input_dir)
            output_file = os.path.join(output_folder, os.path.basename(dicom_path).replace('.dicom', '.png').replace('.dcm', '.png'))
            save_image(image, output_file)

