import argparse
import pandas as pd

def main():
    """Main function to run data analysis of poisonous/edible mushrooms.

       Loads in data, performs exploratory data analysis and preparation/cleaning 
       and finally trains and tests classification algorithm for poisonous/edible 
       mushrooms.

       XGBoost and Random Forest classifiers.

       Run from the command line.
    """
    print("In the main function")
    # Parse command line args for file paths
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
    print(f"input_folder is {input_folder}")

    # data load in
    from data_loading_writing import load_data
    train_df, test_df = load_data(input_folder, train_data_file_name, test_data_file_name)
    
    # rewrite the data to the output folder for reference
    from data_loading_writing import write_data
    write_data(train_df, test_df, output_folder, train_data_file_name, test_data_file_name)


if(__name__ == "__main__"):
    main()