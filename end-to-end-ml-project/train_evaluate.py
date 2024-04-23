
import main_pipeline 
from sklearn.pipeline import make_pipeline 
processing_ = main_pipeline.processing 
import numpy as np
# train 
from sklearn.linear_model import LinearRegression 
# remember lin_reg inherit  , arrtibutes of last  estimator  , such as linear regression has fit , fit_trasnform , predict thus drieclty these arritbutes will be inhreited 

lin_reg = make_pipeline(processing_ , LinearRegression())

# load datasets
xlabels = main_pipeline.df_X
ylabels = main_pipeline.df_Y
df = main_pipeline.df

lin_reg.fit(xlabels , ylabels)
ypred = lin_reg.predict(df)


print(ypred[:4])
print("\n\n")

print(ylabels.iloc[:4])

####################################### check accuracy ##################################
from sklearn.metrics import mean_squared_error  as mse 
lin_mse = mse(np.asarray(ylabels) , np.asarray(ypred)  , squared= False)

print("\n\n\n Accuracy is " ,lin_mse )
print(ypred)

###################### using decision trees

from sklearn.tree import DecisionTreeRegressor
tree_reg = make_pipeline(processing_ , DecisionTreeRegressor(random_state= 42))
tree_reg.fit(xlabels , ylabels)
ypred_tree = tree_reg.predict(xlabels)
print("\n\n Decission Tree regression  " ,  mse(ypred_tree , ylabels))

## this will say mean square error is zero which means , our model has been over fitted  , 