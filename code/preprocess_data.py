# encode catagorical data there are no ordinal relationships so we should use one-hot encoding
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def preprocess_features(df):
    """Preprocesses feature data by scaling numerical features and 
       binary one-hot encoding catagorical features (i.e. non-ordinal catagories).

       Args: df: pandas dataframe: dataframe to preprocess         
    """

    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    scaler = StandardScaler()
    df[numerical_features]=scaler.fit_transform(df[numerical_features])
    
    df=pd.get_dummies(df, columns=categorical_features, drop_first=True)
    one_hot_columns=[col for col in df.columns if col not in numerical_features]
    df[one_hot_columns] = df[one_hot_columns].astype(int)
    return df

def preprocess_target(target_series):
    """Preprocesses target data by encoding catagorical labels as integers.
    """
    label_encoder=LabelEncoder()
    label_encoder.fit_transform(target_series)
    return target_series
