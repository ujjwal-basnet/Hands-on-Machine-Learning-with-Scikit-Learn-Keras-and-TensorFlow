#eda pkgs 
import numpy as np 
import pandas as pd 

#visulization pkgs 
import seaborn  as sns 
import matplotlib.pyplot as plt 

#others 

from pathlib import Path 
import tarfile 
import urllib.request
import os 



#data 
def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents = True  , exist_ok = True )
        url  = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url,tarball_path)
        with tarfile.open(tarball_path , 'r') as housing_tarball:
            housing_tarball.extractall(path = "datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))
        
        






