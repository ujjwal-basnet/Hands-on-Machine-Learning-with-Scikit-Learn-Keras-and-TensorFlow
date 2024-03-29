import basic 
import pandas as pd 
import numpy as np 

#custom transformer class 
from sklearn.base import BaseEstimator , TransformerMixin 

#scalling pkgs 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LinearRegression 



df = basic.load_housing_data()
# print(df.head())

def min_max_scalar(dataset):
    assert not dataset.empty
    dataset = dataset.dropna()
    min_max_scalar = MinMaxScaler(feature_range = (-1 , 1 ))
    min_max_scaled_df = min_max_scalar.fit_transform(dataset.select_dtypes(include = ['number']))
    min_max_scaled_df = basic.pd.DataFrame(data = min_max_scaled_df  , columns= [min_max_scalar.get_feature_names_out()])
    return min_max_scaled_df
    

def stander_scale(dataset):
    assert not dataset.empty 
    dataset = dataset.dropna()
    stander_scale  = StandardScaler()
    stander_scaled_df = stander_scale.fit_transform(dateset.select_dtypes(include = ['number']))
    stander_scaled_df = basic.pd.DataFrame(stander_scaled_df , columns = [stander_scale.get_feature_names_out] ) 
    
    return stander_scaled_df


def log_scale(feature):
    return  feature.apply(basic.np.log)


def inverse_transform(feature):
    target_scalar = StandardScaler()
    scaled_labels = target_scalar.fit_transform(feature.to_frame)
    
    
    model = LinearRegression()
    model.fit(feature[['median_income']] , scaled_labels)
    


class Custom_Transformer(BaseEstimator , TransformerMixin ):
    def __init__(self , column_name):
        self.column_name = column_name 
        
    def fit(self , X , y = None ):
        return self 
    
    def transform(self , X ):
        X_transformed = X.copy()
        X_transformed[self.column_name] = X_transformed[self.column_name].apply(lambda X : X * 2   )
        return X_transformed
    
    

    
data = {'f1': [1 , 2 ,3 ,4 ,5 ] ,
            'f2':[6 , 7 ,8 ,9 ,10]}
df = pd.DataFrame(data)
custom_Transformer = Custom_Transformer(column_name= 'f1')
df_transformed = custom_Transformer.fit_transform(df)
print(df_transformed)
print('\n\n')
print(df)

    




        
        
    


# we can run the following commands 
    

# print('\n')
# print(min_max_scalar(df))



