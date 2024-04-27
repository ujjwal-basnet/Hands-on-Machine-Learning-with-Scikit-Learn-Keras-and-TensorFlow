


import numpy as np 

# load datasets 
data = np.load("/home/ujjwal/cooding/github-p/Hands-on-Machine-Learning-with-Scikit-Learn-Keras-and-TensorFlow/classification/mnist_data.npz" , allow_pickle= True )

# train test 
X  = data['x']
y = data['y']
Xtrain , Xtest  , ytrain , ytest =  X[:600] , X[600:] , y[:600] ,y[600:] 



from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone 
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state= 42)
random_index = np.random.randint(10) 
ytrain_random = (ytrain == str(random_index))
ytest_random  = (ytest == str(random_index))    # generating random index  
skfold = StratifiedKFold(n_splits= 3)
for train_index , testindex in skfold.split(Xtrain , ytrain_random ) :

    clone_clf = clone(sgd_clf)
    xtrainfolods = Xtrain[train_index]
    xtestfolds = Xtest[testindex]
    ytrainfolods = ytrain[train_index]
    ytestfolds = ytest[testindex]

    clone_clf.fit(xtrainfolods , ytrainfolods)
    ypred = clone_clf.predict(xtestfolds)
    n_correct = sum(ypred ==ytestfolds)
    print(n_correct / len(ypred)) # print in percentange like 

    
