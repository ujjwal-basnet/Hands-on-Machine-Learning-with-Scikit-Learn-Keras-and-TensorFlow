import pandas as pd 
import numpy as np 
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.base import BaseEstimator  ,  TransformerMixin


class ClusterSimilarity(BaseEstimator , TransformerMixin):
    def __init__(self , n_clusters  = 10 , gamma = 1.0 , random_state = 42 ): 
        self.n_clusters = n_clusters 
        self.gamma = gamma 
        self.random_state = random_state 

    def fit(self , X , y = None , sample_weight = None ) :

        self.KMeans_ = KMeans(self.n_clusters  , random_state= self.random_state)
        self.KMeans_.fit(X , sample_weight = sample_weight)

        return self  
    

    def transform(self , X ) :
        return rbf_kernel(X , self.KMeans_.cluster_centers_, gamma=self.gamma)


####################### testing###############################


df =  pd.read_csv('/home/ujjwal/cooding/github-p/Hands-on-Machine-Learning-with-Scikit-Learn-Keras-and-TensorFlow/end-to-end-ml-project/datasets/housing.csv')

print("\n This is origin datasets of total rooms")
cluster_simil = ClusterSimilarity( n_clusters  = 10 , gamma = 1.0 , random_state = 42 )
similarities = cluster_simil.fit_transform(df[["latitude", "longitude"]],
                            )


print(similarities.round(2))