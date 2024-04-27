
# train  , test datasets 
""" 
 its good that for mnist datasts , train and test datesets werre sepereated  , 
 also these tranning datasets are already shuffled ,  

 but while working with others data sets we have to shuffled data sets , like what is 
 the digit '5' not included on our test datasets , but only on train datasets , model  might not perform like we want
 right 
    """
import numpy as np 
data = np.load('mnist_data.npz' , allow_pickle= True)
X = data['x']
y = data['y']

Xtrain , Xtest , ytrain , ytest  = X[:600] , X[600:] , y[:600] , y[600:]
print("check the train test split " , Xtrain.shape  , Xtest.shape , ytrain.shape , ytest.shape)

#train SDF cuz, this handes larger datasets good , it deals with tranning data independaintly 
from sklearn.linear_model import SGDClassifier 
sgd_ = SGDClassifier(random_state= 42)

sgd_.fit(Xtrain , ytrain)
random_index = np.random.randint(len(X) - 1 )
pred_value = sgd_.predict([X[random_index]])
if y[random_index] ==pred_value :
    print("yes Correct Prediction")
else :
    print("No , Failled to Predict")

 ###### evaluating using cross validation 
from sklearn.model_selection import cross_val_score 
cross_val_score_  = cross_val_score(sgd_ , Xtrain , ytrain , cv = 30 , scoring='accuracy')
print(cross_val_score_)
# best parameter 




