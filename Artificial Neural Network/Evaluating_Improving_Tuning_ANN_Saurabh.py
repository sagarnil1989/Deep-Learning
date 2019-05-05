#Evaluating, Improving, Tuning Artificial Neural Network

# Part 1 - Data Preprocessing
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Set working directory & import data
dataset = pd.read_csv("Churn_Modelling.csv")
X=dataset.iloc[:,3:13].values
Y=dataset.iloc[:,13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
#To avoid dummy variable trap, we have removed the first column of X
X = X[:, 1:]

# Spliting the dataset into Training set & Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Feature Scaling (to make the features on same scale)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Part 2 - making the ANN!
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential     #Initialize the ANN
from keras.layers import Dense          #builds layers of ANN

# Initialising the ANN
classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
#input_dim is for number of features we have. It will be used as input to the ANN. Mandatory parameter for first hidden layer.
#units is number of units in the layer. Here it is decided by (no. of nodes in input layer + no. of nodes in output layer)/2 = (11+1)/2 = 6
#kernel_initializer = 'uniform' initializes the weight uniformly

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

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
classifier.fit(X_train, Y_train, batch_size = 10, epochs = 100)
#batch_size is no. of observations/rows after which we are adjusting the weight
#One epoch = one Pass (forward & backward) through the ALgo or ANN
#One epoch has multiple iterations if batch size is defined.

#Predicting the Test set result
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Predicting a single new observation
"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""

new_prediction = classifier.predict(sc_X.transform(np.array([[0,0, 600, 1, 40,3,60000,2,1,1,50000]])))
new_prediction = (new_prediction > 0.5)
#Put the customer info in horizontal vector by using double pair of bracket [[]]. Note single [] is for row and [[]] is for column
#Put new info in same order as in the old info on which model is built and it should be in array ie. np.array
#for France dummy variable is 0,0 and for female is 0.
#do the feature scaleing of the horizontal vector befor predicting

#Making the Confusion Matrix to evaluate the prediction
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

#Part 4 - Evaluating, Improving and Tuning the ANN
#Evaluating the ANN using Cross Validation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential     #Initialize the ANN
from keras.layers import Dense          #builds layers of ANN

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 10)
mean = accuracies.mean()
variance = accuracies.std()


#Improving the ANN
# Dropout Regularization to reduce overfitting if needed
#Dropout disables some neurons randomly at each iterations to avoid overfitting
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
import keras
from keras.models import Sequential     #Initialize the ANN
from keras.layers import Dense
from keras.layers import Dropout

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dropout(p = 0.1))        #adding Dropout to 1st hidden layer
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(p = 0.1))        #adding Dropout to 2nd hidden layer
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 10)
mean = accuracies.mean()
variance = accuracies.std()


#Parameter Tuning the ANN using Grid Search
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import keras
from keras.models import Sequential     #Initialize the ANN
from keras.layers import Dense

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100,500],
              'optimizer': ['adam', 'rmsprop']}
#batch_size, epochs etc. parameter name must be same as in the function definition in documentation
#to tune optimizer,loss, units etc., we have to make a parameter in the function build_classifier as done for optimizer 

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, Y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

