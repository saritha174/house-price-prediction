import pandas as pd

#load data
def load_data():
    df = pd.read_csv(r'data\housing.csv')
    return df


#preprocess the data
def preprocess_data(df):
    # remove missing values
    df = df.dropna()

    # convert categorical column to numeric
    df = pd.get_dummies(df, columns=["ocean_proximity"])

    # separate features and target
    X = df.drop("median_house_value", axis=1)

    y = df["median_house_value"]

    return X, y



