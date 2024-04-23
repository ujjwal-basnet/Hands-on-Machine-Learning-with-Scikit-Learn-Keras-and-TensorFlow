from sklearn.pipeline import Pipeline 
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np 


pipeline = Pipeline([
        ('impute' , SimpleImputer(strategy='median')) ,
        ('standerdize' , StandardScaler())
])

# or you can use make make pipeling where you don't need to use the shit , 


from sklearn.pipeline import make_pipeline 
pipeline_ = make_pipeline([
    (SimpleImputer(strategy  = 'medium' )),
    (StandardScaler())
     
      ])
################### testing #################################


df =  pd.read_csv('/home/ujjwal/cooding/github-p/Hands-on-Machine-Learning-with-Scikit-Learn-Keras-and-TensorFlow/end-to-end-ml-project/datasets/housing.csv')
print(df.head())
df_num = df.select_dtypes(include=np.number)
print("\n")
print("\n")


################### transform ####################
df_num_process = pipeline.fit_transform(df_num)
print("numerical only features are ")
print(df_num.head())
print("new numeric only columns is ")
print(df_num.shape)
print("\n\n")

print("Processed Features are")
print(df_num_process[:5][:])
#################### converting them into pandas 

df_num_process = pd.DataFrame(df_num_process , columns= pipeline.get_feature_names_out())\

print("\n\n")
print(df_num_process.head())