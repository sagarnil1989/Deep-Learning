# Self Organizing Map

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, 0:15].values
Y = dataset.iloc[:, -1].values

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
X = sc.fit_transform(X)

#Training the SOM
#Keep minisom.py file in same folder where this SOM file is present
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

#Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    #i is index & x is vector of customers
    w = som.winner(x)
    plot(w[0] + 0.5, w[1] + 0.5, 
         markers[Y[i]], 
         markeredgecolor = colors[Y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()
#w[0] & w[1] are the coordinates of wining nodes, 0.5 is added to put them in center

# Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(1,3)], mappings[(8,3)]), axis = 0)
#(8,1),(6,8) are the coordinates of Outlieing winning node or Outlieing Best Matching Unit (BMU) which will be in white color in visualization plot
#axis = 0 means concatenate along vertical axis
frauds = sc.inverse_transform(frauds)    #inverse the feature scaling