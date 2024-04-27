# impotant notice ,  inorder to run the plots you should have  installed tkinter , i dont know much about why , 
# but i was running on wsl , ubantu  on my windows , when i was printing  ,  i could not see the my plots  , 
# the do couple of searches on internt   , then downloade tkinter  through ubantu apt package manager and it starts showing me plots 

import matplotlib.pyplot as plt 
import numpy as np 

def plot_digit(image_data):
    image = image_data.reshape(28,28)
    plt.imshow(image , cmap='binary')
    plt.axis('off')

     ## Load the data from the saved file
data = np.load("mnist_data.npz" , allow_pickle=True) 

# Extract the arrays x and y
X = data['x']
y = data['y']
some_digit = X[1]
plot_digit(some_digit)


plt.show()
