# Mega Case Study - Make a Hybrid Deep Learning Model (Unsupervised + Supervised)

# Part 1 - Identify the Frauds with the Self-Organizing Map
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
frauds = np.concatenate((mappings[(3,4)], mappings[(6,5)]), axis = 0)
#(8,1),(6,8) are the coordinates of Outlieing winning node or Outlieing Best Matching Unit (BMU) which will be in white color in visualization plot
#axis = 0 means concatenate along vertical axis
frauds = sc.inverse_transform(frauds)    #inverse the feature scaling


# Part 2 - Going from Unsupervised to Supervised Deep Learning

# Creating the matrix of features (independent variables)
customers = dataset.iloc[:, 1:].values   #since customer id will not help in prediction of fraud customers

# Creating the dependent variable
is_fraud = np.zeros(len(dataset))
#is_fraud is dependent variable
for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        #i is for the ith customer (row) and 0 is for customer id ie. 0th coloumn
        is_fraud[i] = 1
        
#Feature Scaling (to make the features on same scale)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
customers = sc_X.fit_transform(customers)

# Part 2 - making the ANN!
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential     #Initialize the ANN
from keras.layers import Dense          #builds layers of ANN

# Initialising the ANN
classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15))
#input_dim is for number of features we have. It will be used as input to the ANN. Mandatory parameter for first hidden layer.
#units is number of neuron units in the layer. Here it is decided by (no. of nodes in input layer + no. of nodes in output layer)/2 = (11+1)/2 = 6
#kernel_initializer = 'uniform' initializes the weight uniformly

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
#activation function for binary output is 'sigmoid', for higher no. of output is 'softmax'

# Compiling the ANN ie. adding Stochastic Gradient Descent  to the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#optimizer is algorith to optimize the weights during Back-propagation. 'adam' is one of the algo of SGD
#In adam (Adaptive Optimizer), initally alpha (learning rate) is high and gradually decreases with every epoch, where as the normal SGD alpha remains constant
#loss = 'binary_crossentropy' for two output of dependent variable and 'categorical_crossentropy' for more than 2 output
#metrics is criteria to improve the ANN model performance

# Fitting the ANN to the Training set
classifier.fit(customers, is_fraud, batch_size = 1, epochs = 2)
#customers is independent variables, is_fraud is dependent variable
#batch_size is no. of observations/rows after which we are adjusting the weight
#One epoch = one Pass (forward & backward) through the ALgo or ANN
#One epoch has multiple iterations if batch size is defined.

# Predicting the probabilities of frauds
y_pred = classifier.predict(customers)     #predicted probabilities
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis = 1)   #axis = 1 means concatenate along horizontal axis
#concatenation of customer ids and predicted probabilities
y_pred = y_pred[y_pred[:, 1].argsort()]
#argsort() will sort the y_pred array on the basis of predicted probabilities which is column no. 1 in python ie. y_pred[:, 1]