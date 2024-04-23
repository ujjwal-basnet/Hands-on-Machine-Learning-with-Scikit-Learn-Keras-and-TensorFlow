
##taking about challenge about pipeline 3 :
    # - listing all the columns names was not very convienient 
    


# here we are using make ....  like make pipeline , make colum tranformer , in make we dont have to specify names

import pandas as pd 
import numpy as np 

from sklearn.compose import make_column_selector , make_column_transformer 
from sklearn.pipeline import make_pipeline 
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.preprocessing import StandardScaler




num_pipeline =  make_pipeline(
    SimpleImputer(strategy= 'median') , 
    StandardScaler()

)


cat_pipeline = make_pipeline(
    SimpleImputer(strategy='most_frequent') , 
    OneHotEncoder(handle_unknown = 'ignore')
)

processing = make_column_transformer(

    (num_pipeline ,  make_column_selector(dtype_include=np.number)  )  ,  
    (cat_pipeline  ,  make_column_selector(dtype_include= object))
)


df =  pd.read_csv('/home/ujjwal/cooding/github-p/Hands-on-Machine-Learning-with-Scikit-Learn-Keras-and-TensorFlow/end-to-end-ml-project/datasets/housing.csv')
df_X = df.iloc[: , list(range(1  , 8)) + list(range(9 , 10))]

X_process = processing.fit_transform(df_X)
X_process = pd.DataFrame(X_process , columns= processing.get_feature_names_out())
print(X_process.head())

