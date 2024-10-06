import argparse
import pandas as pd

def main():
    """Main function to run data analysis of poisonous/edible mushrooms.

       Loads in data, performs preparation/cleaning 
       and finally trains, tests and tunes classification algorithms 
       for poisonous/edible mushrooms - XGBoost and Random Forest 
       classifiers.

       Run from the command line.
    """
    print("Main function:")
    # parse command line args for file paths
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder")
    parser.add_argument("--train_data_file_name")
    parser.add_argument("--test_data_file_name")
    parser.add_argument("--output_folder")
    args = parser.parse_args()
    input_folder = args.input_folder
    train_data_file_name = args.train_data_file_name
    test_data_file_name = args.test_data_file_name
    output_folder = args.output_folder
    print(f"Path to input_folder is {input_folder}")
    print(f"Path to output_folder is {output_folder}")

    # data load in
    from load_write_data import load_data
    train_df, test_df = load_data(input_folder, train_data_file_name, 
                                  test_data_file_name)
    
    # rewrite the raw data to the output folder for reference
    from load_write_data import write_data
    write_data(train_df, test_df, output_folder, 
               train_data_file_name, test_data_file_name)
    
    # split features and target (in this case 'class' column) from the data
    from load_write_data import features_target_split
    features_df, target_series = features_target_split(train_df, 'class')
    
    # data cleaning
    from clean_data import cleaner
    features_df = cleaner(features_df, df_key="features_df")
    test_df = cleaner(test_df, df_key="test_df")

    # data preprocessing
    from preprocess_data import preprocess_features, preprocess_target
    X = preprocess_features(features_df)
    y = preprocess_target(target_series)
    test_df = preprocess_features(test_df)

    # 

    # K-fold Cross Validation model training and validation


if(__name__ == "__main__"):
    main()