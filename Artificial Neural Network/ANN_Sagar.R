#Artificial Neural Network

#Importing data
dataset = read.csv('Churn_Modelling.csv')
dataset = dataset[ , 4:14]

#Encoding categorical data
dataset$Geography = as.numeric(factor(dataset$Geography, levels = c('France', 'Spain', 'Germany'), 
                         labels = c(1,2,3)))

dataset$Gender = as.numeric(factor(dataset$Gender, levels = c('Female', 'Male'), 
                           labels = c(1,2)))

# Spliting the dataset into Training set & Test set
#install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Exited, SplitRatio = 0.8)
training_set = subset(dataset, split == T)
test_set = subset(dataset, split == F)

#Feature Scaling
training_set[ , -11] = scale(training_set[ , -11])
test_set[ , -11] = scale(test_set[ , -11])


# Fitting ANN to the Training set
#install.packages('h2o')
library(h2o)
h2o.init(nthreads = -1)
classifier = h2o.deeplearning(y = 'Exited',
                         training_frame = as.h2o(training_set),
                         activation = 'Rectifier',
                         hidden = c(6,6),
                         epochs = 100,
                         train_samples_per_iteration = -2)

#hidden is for hidden layers, c(6,6) means 2 hidden layer with 6 neurons in both layer
#train_samples_per_iteration = -2 helps in auto tune the batch size

# Predicting the Test set results
y_pred = h2o.predict(classifier, newdata = as.h2o(test_set[-11]))
y_pred = (y_pred > 0.5)
y_pred = as.vector(y_pred)

# Making the Confusion Matrix
cm = table(test_set[, 11], y_pred)

h2o.shutdown()