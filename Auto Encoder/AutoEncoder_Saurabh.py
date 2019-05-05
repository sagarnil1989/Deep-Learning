#Auto Encoder

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn               #module of torch to implement neural network
import torch.nn.parallel            #module of torch to implement Parallel computation
import torch.optim as optim         #module of torch to implement Optimizer
import torch.utils.data
from torch.autograd import Variable  #module of torch to implement Stochastic Gradient Descent

#Go to https://grouplens.org/datasets/movielens/ to download the dataset
#Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
#all the files are present under ml-1m
#columns are seperated by :: in data and there is no column name / header present in the data
#encoding is the laguages used in data. English & Latin both are used.

# Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')        #tab ie. '\t' must be specified with delimeter, not with seperator ie. sep
training_set = np.array(training_set, dtype = 'int')            #For torch library to use, we have to convert the data frame to arrays of integer

test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')        #tab ie. '\t' must be specified with delimeter, not with seperator ie. sep
test_set = np.array(test_set, dtype = 'int')            #For torch library to use, we have to convert the data frame to arrays of integer

#Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))     #0th coloumn has user id
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))    #1st column has movie id

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    #data can be training or test set
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        #data[:,1] is movie id column, data[:,0] is user id column. data[:,0] == id_users is the user id of a one particular user (in 1st iteration, user id 1)
        #data[:,1] [data[:,0] == id_users]  is movie ids seen/rated by a particular user id. In 1st iteration we will get all movie ids rated by user id 1.
        #id_movies has the movie ids rated by an user id
        id_ratings = data[:,2][data[:,0] == id_users]
        #data[:,2] is rating value column. data[:,0] == id_users is the user id of a one particular user (in 1st iteration, user id 1)
        #data[:,2] [data[:,0] == id_users] is rating value given by a particular user id. In 1st iteration we will get all rating values rated by user id 1.
        #id_ratings has the rating value given by an user id
        ratings = np.zeros(nb_movies)       #initialised the rating value of all movies to zero in a list data type.
        ratings[id_movies - 1] = id_ratings #combining movie id (it starts with 1) and its rating value. The movies which has not been rated by a particular user (in 1st iteration, user id 1) will have rating value zero, since rating value for all movies is already initialised to zero for the particular user id.  
        new_data.append(list(ratings))      #Creates list of list ie. list of rating value for a particular user id (ratings) is kept inside a new_data list
    return new_data
    #new_data list will contain 943 list (total no. of users) having rating value, each list of 1682 size (total no. of movies).
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)
#In spyder's Variable explorer training_set & test_set will dissapear because pytorch came recently and not recognised by the Spyder yet.

# Creating the architecture of the Neural Network
class SAE(nn.Module):
    #SAE is Stacked Auto Encoder
    #Inheriting class 'Module' from 'nn' library
    def __init__(self, ):
        super(SAE, self).__init__()
        #using 'super' we will get all the class & methods of parent module nn
        self.fc1 = nn.Linear(nb_movies, 20)      #fc1 is 1st full connection #nb_movies is visible input layer and 20 neurons in first hidden layers
        self.fc2 = nn.Linear(20, 10)             #20 neuron in first hidden layer and 10 neuron in second hidden layer
        self.fc3 = nn.Linear(10, 20)             #10 neuron in second hidden layer and 20 neuron in third hidden layer
        self.fc4 = nn.Linear(20, nb_movies)      #20 neuron in third hidden layer, nb_movies in visble output layer 
        self.activation = nn.Sigmoid()           #activation function is sigmoid from nn library

    def forward(self, x):
        x = self.activation(self.fc1(x))         #Encoding
        x = self.activation(self.fc2(x))         #Encoding
        x = self.activation(self.fc3(x))         #Decoding
        x = self.fc4(x)
        return x
sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)

# Training the SAE
nb_epoch = 200       #no. of epoch
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.      #float variable for no. of users who have rated atleat 1 movie. float variable because we will calculate RMSE at the end which requires all float values.
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0)    #batch learning
        target = input.clone()          #target is clone of input ie. both are same
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            target.require_grad = False      #SGD is not calculated wrt target
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()      #direction in which weights will be updated
            train_loss += np.sqrt(loss.data[0]*mean_corrector)
            s += 1.
            optimizer.step()     #intensity by which weights will be updated
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

# Testing the SAE
test_loss = 0
s = 0.
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user])
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data[0]*mean_corrector)
        s += 1.
print('test loss: '+str(test_loss/s))