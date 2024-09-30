# Data load in from input folder and data writing to out put folder functions

import pandas as pd
import os

def load_data(input_folder, train_data_file_name, test_data_file_name):
    train_df = pd.read_csv(os.path.join(input_folder, train_data_file_name))
    test_df = pd.read_csv(os.path.join(input_folder, test_data_file_name))
    return train_df, test_df

def write_data(train_df, test_df, output_folder, train_data_file_name, 
               test_data_file_name):
    train_df.to_csv(os.path.join(output_folder,train_data_file_name))
    test_df.to_csv(os.path.join(output_folder,test_data_file_name))
    return None