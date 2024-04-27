## this is introduction about minist and skleaan load  module 

# featching minist datasets from open ml 
from sklearn.datasets import fetch_openml 
mnist = fetch_openml('mnist_784' , as_frame=False)
import numpy as np 

## sklearn datasts mostly contains 3 features 
## fetch* to downlode from web 
## load_*  to load offline pre downloaded datasets from sklearn 
## make_*  to generate fake datasets , 


## most datasets are return as numpy arrs  ,  but some are return as [sklearn.utils.Bunch Objects] which are dictinory
## whose entities can be acced via attributes like 
"""
         descr : for discription of datasets 
         data : return input data , usally 2d numpy array 
         target : return label usally , 1d numpy array 
        
"""

X, y = mnist.data , mnist.target 
# target
print("This is input data  \n ",X.shape)
print("\n" , X[:5])
# traning
print("\n This is label  \n" , y.shape)
print("\n " ,  y[:5])


# # save 
# np.savez("mnist_data.npz", x = X , y = y ) 
"""  inorder to retrive
     Load the data from the saved file
data = np.load("mnist_data.npz")

# Extract the arrays x and y
x = data['x']
y = data['y']

    """