import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector , make_column_transformer 

def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy='median'),
        FunctionTransformer(func=column_ratio, validate=False),
        StandardScaler()
    )

log_pipeline = make_pipeline(
    SimpleImputer(strategy='median'),
    FunctionTransformer(np.log, validate=False),
    StandardScaler()
)

default_num_pipeline = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler()
)

cat_pipeline = make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    OneHotEncoder()
)

processing = ColumnTransformer([
    ('bedrooms', ratio_pipeline(), ['total_bedrooms', 'total_rooms']),
    ('rooms_per_house', ratio_pipeline(), ['total_rooms', 'households']),
    ('people_per_house', ratio_pipeline(), ['population', 'households']),
    ('log', log_pipeline, ["total_bedrooms", "total_rooms", "population", "households", "median_income"]),
    ("cat", cat_pipeline, make_column_selector(dtype_include=object))
],
remainder=default_num_pipeline)

# Load data
df = pd.read_csv('/home/ujjwal/cooding/github-p/Hands-on-Machine-Learning-with-Scikit-Learn-Keras-and-TensorFlow/end-to-end-ml-project/datasets/housing.csv')
df_X = df.iloc[:, list(range(1, 8)) + list(range(9, 10))]  # Selecting relevant columns
df_Y = df['median_house_value']
# Apply the transformation
X_process = processing.fit_transform(df_X)

# Convert transformed data to DataFrame for printing
X_process_df = pd.DataFrame(X_process)
print(X_process_df.head())
