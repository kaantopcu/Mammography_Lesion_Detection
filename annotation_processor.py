import pandas as pd

def process_annotations(input_file, output_file):
    """
    Processes the annotation file by adding a unique index and image name column, then saves the updated file.

    :param input_file: Path to the input annotation CSV file.
    :param output_file: Path to save the updated annotation CSV file.
    """
    # Read the annotation file
    df = pd.read_csv(input_file)

    # Add a unique index column
    df.reset_index(inplace=True)  # Adds 'index' column with unique row indices

    # Create 'image_name' column
    df['image_name'] = df['image_id'] + "_" + df['laterality'] + "_" + df['view_position']

    # Save the updated DataFrame to the specified output file
    df.to_csv(output_file, index=False)

    print(f"Updated annotations saved to {output_file}")

# Example usage
input_annotation_file = "annotations.csv"  # Replace with your input file path
output_annotation_file = "updated_annotations.csv"  # Replace with your desired output file path

# Process the annotations
process_annotations(input_annotation_file, output_annotation_file)
