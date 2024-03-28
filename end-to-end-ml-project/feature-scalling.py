import basic 


#scalling pkgs 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import StandardScaler 



df = basic.load_housing_data()
print(df.head())

def min_max_scalar(dataset):
    assert not dataset.empty
    dataset = dataset.dropna()
    min_max_scalar = MinMaxScaler(feature_range = (-1 , 1 ))
    min_max_scaled_df = min_max_scalar.fit_transform(dataset.select_dtypes(include = ['number']))
    min_max_scaled_df = basic.pd.DataFrame(data = min_max_scaled_df  , columns= [min_max_scalar.get_feature_names_out()])
    return min_max_scaled_df
    

def stander_scale(dataset):
    assert not dataset.empty 
    dataset = dataset.dropna()
    stander_scale  = StandardScaler()
    stander_scaled_df = stander_scale.fit_transform(dateset.select_dtypes(include = ['number']))
    stander_scaled_df = basic.pd.DataFrame(stander_scaled_df , columns = [stander_scale.get_feature_names_out] ) 
    
    return stander_scaled_df





# we can run the following commands 
    

# print('\n')
# print(min_max_scalar(df))



