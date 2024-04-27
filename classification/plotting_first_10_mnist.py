# printing first 10 digits in mnist datasts
import numpy as np 
# load numpy data se
 #Load the data from the saved file
data = np.load("mnist_data.npz" , allow_pickle= True )

# Extract the arrays x and y
x = data['x']
y = data['y']

import matplotlib.pyplot as plt 
fig , ax = plt.subplots(figsize = (8 , 4)  , nrows = 2 , ncols= 4 )
temp = 0
for i in range(2):
    for j in range(4):
        
        ax[i][j].imshow(x[temp].reshape(28 , 28 ) , cmap = 'binary')
        ax[i][j].axis('off')
        
        temp = temp + 1 

plt.show()

