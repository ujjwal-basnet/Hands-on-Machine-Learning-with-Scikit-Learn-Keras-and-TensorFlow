
import numpy as np 
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score 

# load datasets 
data = np.load("/home/ujjwal/cooding/github-p/Hands-on-Machine-Learning-with-Scikit-Learn-Keras-and-TensorFlow/classification/mnist_data.npz" , allow_pickle= True )

# train test 
Xtrain , Xtest  , ytrain , ytest =  data['x'][:600] , data['x'][600:] , data['y'][:600] , data['y'][600:] 

#generating 0 elese 1  1-d  array  
ytrain_5 = (ytrain == '5' )
ytest_5  = (ytest == '5') 


#model accuracy test 
print("\n\n SGD Classifier arruracy cross_val_score" , cross_val_score(SGDClassifier() , Xtrain , ytrain_5 , cv = 3 , scoring= 'accuracy' ))
""" 
     you will see somting like 0.955 0.98 0.925  , accuracy . Preety good right ?
     lets see what will be the accuray of dummy classifeer which , classises every single items in  most  , frequent class 
      
        for example let say 5 is most freuqeuncy class , then dummyclassifier will pecited all others number as 5
  
   """


from sklearn.dummy import DummyClassifier 
print("\n\n  Dummy  Classiser accuracy  " , cross_val_score(DummyClassifier(random_state= 42) ,Xtrain , ytrain_5 , cv =3 , scoring= 'accuracy'))

"""      even accuracy of dummy classifier is  -> 0.915 0.915 0.915 
        what  ? how's that even possiable ? heh ?

         look  , 5 is less than 10% in the data sets  ,that means 90 %  will be always false  then if you predict 
         that means if you said 5 is not in datasets , you are 90% correct ,    
            
          """


''' 
    accuracy is not , preferead meause for classifier's , espically when you are dealing with skwed datasets 
       i.e when some classes are most frequent than others  '''

# implementation Cross-Validation
