#handeling numerical and categorical values together 
from sklearn.compose import ColumnTransformer
import pandas as pd 
import numpy as np 
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
    # note that median_house_value is what we are going to predict



df =  pd.read_csv('/home/ujjwal/cooding/github-p/Hands-on-Machine-Learning-with-Scikit-Learn-Keras-and-TensorFlow/end-to-end-ml-project/datasets/housing.csv')

# removing Y feature and only focuing in X 
num_attributes = df.select_dtypes(include=np.number).drop(columns=['median_house_value'])


cat_attributes = df.select_dtypes(include = object)

#categorical pipeline 
cat_pipeline = make_pipeline(
    SimpleImputer(strategy= 'most_frequent'),
        OneHotEncoder(handle_unknown=  "ignore")
)


num_pipeline = make_pipeline(
    
        SimpleImputer(strategy= 'median') , 
        StandardScaler()
    
)

processing  = ColumnTransformer([

    ('num', num_pipeline, num_attributes.columns.tolist() ) ,
    ('cat' , cat_pipeline , cat_attributes.columns.tolist())
])


df_processing = processing.fit_transform(df)

# Print out the columns after transformation

df = pd.DataFrame(df_processing  , columns= processing.get_feature_names_out())
print(df)