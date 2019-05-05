# Boltzmann Machines

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

# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)
training_set[training_set == 0] = -1    #movies which has not been rated by the user had zero value previously which is replaced by -1
training_set[training_set == 1] = 0     # rating value 1 or 2 is not liked by the user, so assigned the value of zero to them. 
training_set[training_set == 2] = 0     #In pytorch or logical operator can't be used like python so wrote two seperate statement for rating 1 & 2 to be zero
training_set[training_set >= 3] = 1     #Rating 3, 4, 5 is assigned to 1

test_set[test_set == 0] = -1    #movies which has not been rated by the user had zero value previously which is replaced by -1
test_set[test_set == 1] = 0     # rating value 1 or 2 is not liked by the user, so assigned the value of zero to them. 
test_set[test_set == 2] = 0     #In pytorch or logical operator can't be used like python so wrote two seperate statement for rating 1 & 2 to be zero
test_set[test_set >= 3] = 1     #Rating 3, 4, 5 is assigned to 1

# Creating the architecture of the Neural Network (Restricte Botzmann Machine)
class RBM():
    def __init__(self, nv, nh):
        #nv is no. of visible nodes and nh is no. of hidden nodes
        self.W = torch.randn(nh, nv)     #W is weight and it is initialised randomly (mean = 0, variance = 1) and is of size (nh,nv)
        self.a = torch.randn(1, nh)      #a is bias for probability of hidden node given visible node ie. P(h/v)
        self.b = torch.randn(1, nv)      #b is bias for probability of visible node given hidden node ie. P(v/h)

    def sample_h(self, x):
        #sampling the hidden node according to the P(h/v) ie probabilty of hidden node given visible node
        #x is visible neuron
        wx = torch.mm(x, self.W.t())
        #mm is for multiplication of 2 tensors ie. Matrix Multiplication
        #W.t() is transpose of W
        activation = wx + self.a.expand_as(wx)          #activation function is weighted input + bias. It represents the probability that hidden node gets activated according to the value of visible node.
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_v(self, y):
        #y is hidden node
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    def train(self, v0, vk, ph0, phk):
        #For Contrastive Divergence
        #v0 = input vector ie. ratings by one user
        #vk = visible node obtained after k sampling ie. after k rounds from visble node to hidden node to visble node inorder to converge
        #ph0 = probability after first iteration of hidden node equal to 1 given the value of v0
        #phk = probabilty of hidden node after k sampling given the value of vk
        self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)

nv = len(training_set[0])
nh = 100
batch_size = 100
rbm = RBM(nv, nh)     #RBM object

# Training the RBM
nb_epoch = 10        #no. of epoch
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.         #float variable for no. of users who have rated atleat 1 movie. float variable because we will calculate RMSE at the end which requires all float values.
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

# Testing the RBM
test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
print('test loss: '+str(test_loss/s))