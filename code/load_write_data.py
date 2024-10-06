# Data load in from input folder and data writing to out put folder functions
import pandas as pd
import os

def load_data(input_folder, train_data_file_name, test_data_file_name, index_col=None):
    """Loads training and test data from input folder and returns as 
    pandas dataframes.

    Args: input_folder: str: path to input folder containing data files
            train_data_file_name: str: name of training data file
            test_data_file_name: str: name of test data file
            index_col: str: name of index column if present 
                            (default=None, no index column)
    """

    train_df = pd.read_csv(os.path.join(input_folder, train_data_file_name), 
                           index_col='id')
    test_df = pd.read_csv(os.path.join(input_folder, test_data_file_name), 
                          index_col='id')
    return train_df, test_df

def write_data(train_df, test_df, output_folder, train_data_file_name, 
               test_data_file_name):
    """Writes training and test data to output folder as csv files.

    Args: train_df: pandas dataframe: training data
            test_df: pandas dataframe: test data
            output_folder: str: path to output folder
            train_data_file_name: str: name of training data file
            test_data_file_name: str: name of test data file 
    """
    train_df.to_csv(os.path.join(output_folder,train_data_file_name))
    test_df.to_csv(os.path.join(output_folder,test_data_file_name))
    return None

def features_target_split(df, target_col):
    """Splits a dataframe into features and target dataframes.

    Args: df: pandas dataframe: dataframe to split
            target_col: str: name of target column
    """        
    target_series = df[target_col]
    features_df = df.drop(target_col, axis=1)
    features=features_df.columns.tolist()
    print(f"target is {target_col}")
    print(f"features are {features}")
    return features_df, target_series