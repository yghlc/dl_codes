
# coding: utf-8

# ## Import numpy and matplotlib library
# 
# A simple example showing how to standardise the data with zero mean and standard variance.

# In[1]:

import numpy as np
import matplotlib.pyplot as plt


# Define the plot function for visualization.

# In[2]:

def plot(x):
    r = 6
    plt.scatter(x[:, 0], x[:, 1], s=5, alpha=0.5)
    plt.ylim([-r, +r])
    plt.xlim([-r, +r])
    plt.grid()
    plt.show()
    


# Generate a dataset with 2000 samples from 2D Gaussian distribution.

# In[3]:

x = np.random.multivariate_normal([2, 2], [[1, 0.1],[0.9, 0.5]], 2000)
print(x.shape)

plot(x)


# Subtract mean to remove the shift of the dataset.

# In[4]:

# normalize by subtracting mean
x -= np.mean(x, axis=0) 
plot(x)


# Devide the standard deviate to remove the scale of the dataset.

# In[5]:

# normalize by divide std
x /= np.std(x, axis=0)
plot(x)

