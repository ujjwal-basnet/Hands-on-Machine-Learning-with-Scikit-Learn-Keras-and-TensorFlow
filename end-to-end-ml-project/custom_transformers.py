from sklearn.preprocessing import FunctionTransformer 
import pandas as pd 
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
# to create a custom transformaers  one need to create , a class that inherit from "BaseEstimator" and "TransformerMixin"
import seaborn as sns

df =  pd.read_csv('/home/ujjwal/cooding/github-p/Hands-on-Machine-Learning-with-Scikit-Learn-Keras-and-TensorFlow/end-to-end-ml-project/datasets/housing.csv')
print("Skwnees before log transformation is "  , df[['population']].skew())

#transforming 
from sklearn.preprocessing import FunctionTransformer 


log_transformer = FunctionTransformer(np.log , inverse_func= np.exp) 
log_pop = log_transformer.transform(df[['population']])


print("Skwnees before log transformation is "  , log_pop.skew())



## 

sf_coords = 37.7749, -122.41
sf_transformer = FunctionTransformer(rbf_kernel,
                                     kw_args=dict(Y=[sf_coords], gamma=0.1))
sf_simil = sf_transformer.transform(df[["latitude", "longitude"]])

print('\n' , '\n')
print(sf_simil)