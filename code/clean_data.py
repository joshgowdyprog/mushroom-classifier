import pandas as pd


def cleaner(df, df_key=None, target_col=None):
    """Cleans train/test/etc. data frame by filling missing catagorical 
    data with 'missing' and replacing low frequency/erroneous catagorical
    labels with 'noise' catagories. Missing numerical data is imputed with
    mean values.

    Args: df: pandas dataframe: dataframe to clean
          df_key: str: name of dataframe
          target_col: str: name of target column (default=None, means we have 
                           no target column e.g. for features only or test 
                           data frames)
    """
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    if target_col in categorical_features: categorical_features.remove(target_col)

    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    for col in categorical_features:
        df[col] = df[col].fillna('missing')
        df.loc[df[col].value_counts(dropna=False)[df[col]].values < 100, col] = "noise"
        df[col] = df[col].astype('category')

    for col in numerical_features:
        df[col] = df[col].fillna(df[col].mean())

    if (df.isnull().sum() == 0).all():
     print(f"No remaining missing values in {df_key} dataframe after cleaning.")

    return df