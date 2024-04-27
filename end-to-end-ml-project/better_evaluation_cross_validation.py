# BETTER USE  SCIKITLEARN K FOLD CROSS-VALIDATION FEATURE  , IT RANDOMS SPLITS THE TRANNING SET INTO 10 N
# NON OVERLAPING DUBDRYD CALLED FOLDS , THEN IT TRAINS AND EVALIATE THE DECISION TREE MODELS 10 TIMES 
# picking diff fold for evaluation  every time  using 9 folds for tranning .


# result is the array contaning the 10 evaluation scores :
import pandas as pd 
from sklearn.model_selection import cross_val_score 
import train_evaluate
from sklearn.pipeline import  make_pipeline  


xlabel= train_evaluate.xlabels
ylabel = train_evaluate.ylabels
df = train_evaluate.df 
ypred_tree = train_evaluate.ypred_tree
tree_reg = train_evaluate.tree_reg

tree_rmse = cross_val_score(tree_reg , X =xlabel  , y =ylabel , cv = 10 )
print(pd.Series(tree_rmse).describe())


############################## see mean and std and compare with previously linear regression  


############## using randomforest  #####################

from sklearn.ensemble import RandomForestRegressor
processing_ = train_evaluate.processing_
forest_reg = make_pipeline(processing_  , RandomForestRegressor())
forest_reg.fit(xlabel , ylabel)
forest_pred = forest_reg.predict(xlabel)


forest_cv = cross_val_score(forest_reg , X = xlabel , y = ylabel , cv = 6 )
print("\n\n\n forest cv is " , forest_cv)

print(pd.Series(forest_cv).describe())
