from sklearn.base import BaseEstimator  ,  TransformerMixin
from sklearn.utils.validation import check_array , check_is_fitted  #contains several  function that we can use to validate the inpus , 
import pandas as pd 
import numpy as np 

# wee need baseestimator and tranformaerMixin to create own custom transformer classes   
#transformermixin is resopnsiable for fit() , transofmr() while another one is responsible for 
# get_params() , set_paramas()


class StandardScalerClone(BaseEstimator , TransformerMixin):
    def __init__(self , with_mean = True ):
        self.with_mean = with_mean
    
    def fit(self , X , y=None  ): #  scikit leanr fit () requires two vairbale X and y ,   but here we are not using y thus None , 
        X = check_array(X) #cheks if X is an array or not with finite float values
        
        ## Standarization is  X - mean / std  so lets caclulate those 

        self.mean_ = X.mean(axis = 0 ) 
        self.scale_ = X.std(axis =  0 )
        self.n_features_in_ = X.shape[1]
        

        return self # always return self 
    


    def transform(self , X ) :# dont use y here 
        check_is_fitted(self)
        X = check_array(X)
        assert self.n_features_in_ == X.shape[1] ## ensuring data passed to transform / predict has same no of feature as fit () or not 
        if self.with_mean:
            X = X - self.mean_ 
        
        return X / self.scale_
    



####################### testing###############################
from sklearn.preprocessing import StandardScaler 

df =  pd.read_csv('/home/ujjwal/cooding/github-p/Hands-on-Machine-Learning-with-Scikit-Learn-Keras-and-TensorFlow/end-to-end-ml-project/datasets/housing.csv')

print("\n This is origin datasets of total rooms")
print("\n" ,  df[['total_rooms']])

print("\n Using Scikitlearn Stander scale ")
print("\n " ,StandardScaler().fit_transform(df[['total_rooms']]) )


print("\n using our clone ") 
print("\n"  , StandardScalerClone().fit_transform(df[['total_rooms']]))