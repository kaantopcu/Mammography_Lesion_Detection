import pandas as pd
from sklearn.model_selection import train_test_split

def create_stratify_column(df, columns):
    """
    Create a stratify column by combining the values from the specified columns.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        columns (list): List of column names to be combined for stratification.
    
    Returns:
        pd.DataFrame: DataFrame with a new stratify column.
    """
    df['stratify_column'] = df[columns].agg('_'.join, axis=1)
    return df

def filter_low_freq_classes(df, stratify_column, min_samples=2):
    """
    Filter out rows with low-frequency classes for stratification.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        stratify_column (str): The name of the stratify column.
        min_samples (int): The minimum number of samples for a class to be considered.
    
    Returns:
        pd.DataFrame: Filtered dataframe with low-frequency classes excluded.
    """
    class_counts = df[stratify_column].value_counts()
    low_freq_classes = class_counts[class_counts < min_samples].index
    df.loc[df[stratify_column].isin(low_freq_classes), 'split'] = 'None'
    return df[df['split'] != 'None']

def split_data(df, stratify_column, test_size=0.2, random_state=42):
    """
    Perform a stratified split of the data into training and validation sets.
    
    Args:
        df (pd.DataFrame): The dataframe to split.
        stratify_column (str): The column used for stratification.
        test_size (float): The proportion of data to use for validation.
        random_state (int): The random seed for reproducibility.
    
    Returns:
        pd.DataFrame: DataFrame with updated split information.
    """
    train_set, val_set = train_test_split(
        df, test_size=test_size, stratify=df[stratify_column], random_state=random_state
    )
    train_set['split'] = 'training'
    val_set['split'] = 'validation'
    return pd.concat([train_set, val_set])

def combine_and_save(df, original_df, output_file='train_valid_test_split.csv'):
    """
    Combine the split data with the original dataset and save it to a CSV file.
    
    Args:
        df (pd.DataFrame): The split dataframe.
        original_df (pd.DataFrame): The original dataframe with non-split data.
        output_file (str): The name of the output CSV file.
    
    Returns:
        pd.DataFrame: Combined dataframe.
    """
    columns_to_check = original_df.columns.tolist()
    columns_to_check.remove('split')  # Exclude the `split` column

    # Combine the datasets and drop duplicates
    combined_df = pd.concat([df, original_df]).drop_duplicates(subset=columns_to_check, keep='first')
    combined_df.to_csv(output_file, index=False)
    return combined_df

# Main function to run the entire workflow
def split_and_save_data(train_df, columns_to_stratify, output_file='train_valid_test_split.csv'):
    """
    Perform the entire split and save the result to a CSV.
    
    Args:
        train_df (pd.DataFrame): The dataframe containing the raw data.
        columns_to_stratify (list): List of column names for stratification.
        output_file (str): The output file name.
    
    Returns:
        pd.DataFrame: The final combined and processed dataframe.
    """
    # Step 1: Create the stratify column
    train_df = create_stratify_column(train_df, columns_to_stratify)

    # Step 2: Filter out low-frequency classes
    train_data = train_df[train_df['split'] == 'training']
    train_data = filter_low_freq_classes(train_data, 'stratify_column')

    # Step 3: Perform the stratified split
    split_df = split_data(train_data, 'stratify_column')

    # Step 4: Combine with the testing data and save to CSV
    return combine_and_save(split_df, train_df, output_file)
