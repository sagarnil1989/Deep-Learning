# Convolutional Neural Network

# Part 1 - Building the CNN
# Importing the Keras libraries and packages
from keras.models import Sequential        #Initialize the Neural Network
from keras.layers import Conv2D            #To add convolutional layers in CNN for images which are in 2D
from keras.layers import MaxPooling2D      #To add pooling layers
from keras.layers import Flatten           #Convert pooled feature maps into large feature vector that acts as input to Neural network
from keras.layers import Dense             #builds layers of Neural Network

# Initialising the CNN
classifier = Sequential()

#4 steps of CNN
# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
#32 feature detectors or filters or kernel of dimension 3*3
#default size of stride = 1
#input_shape is expected format of input images ie. the format in which input images will be converted. Dimension of 2D array = 64*64, no. of channels = 3(Red, Green, Blue)

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#Max pooling reduces the size of Feature map without losing its performance/features.
#default value of stride = pool_size

# Adding a second convolutional layer will improve the accuracy of prediction of CNN
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))        #hidden layer
classifier.add(Dense(units = 1, activation = 'sigmoid'))       #output layer
#units is number of neuron units in the layer.
#activation = 'sigmoid' for binary output, activation = 'softmax' for multiple output

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#optimizer is algorith to optimize the weights during Back-propagation. 'adam' is one of the algo of SGD
#In adam (Adaptive Optimizer), initally alpha (learning rate) is high and gradually decreases with every epoch, where as the normal SGD alpha remains constant
#loss = 'binary_crossentropy' for two output of dependent variable and 'categorical_crossentropy' for more than 2 output
#metrics is criteria to evaluate the ANN model performance

# Part 2 - Fitting the CNN to the images
# Generating batches of tensor image data with real-time data augmentation. The data will be looped over (in batches) indefinitely.
#Go to https://keras.io/preprocessing/image and copy the code of Example of using .flow_from_directory(directory) section and change it as per requirement.
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
#this is the augmentation configuration we will use for training
#rescale is a value by which we will multiply the data before any other processing. Our original images consist in RGB coefficients in the 0-255, but such values would be too high for our models to process (given a typical learning rate), so we target values between 0 and 1 instead by scaling with a 1/255
#shear_range is displacing each pixel of image in a fixed direction ie. geometrical transformation
#zoom_range is for randomly zooming inside pictures
#horizontal_flip is for randomly flipping half of the images horizontally --relevant when there are no assumptions of horizontal assymetry (e.g. real-world pictures).

test_datagen = ImageDataGenerator(rescale = 1./255)
# this is the augmentation configuration we will use for testing:only rescaling

# this is a generator that will read pictures found in subfolers of 'dataset/training_set', and indefinitely generate batches of augmented image data
#Increasing the target_size pixel tp 128 or 1024 will improve the CNN prediction accuracy
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
# all images will be resized to 64*64 which is same as input_shape in Step 1 - Convolution
#batch_size is no. of images after which we are adjusting the weight
#since we use binary_crossentropy loss in Compiling the CNN step, so we need class_mode = 'binary' 

#Increasing the target_size pixel tp 128 or 1024 will improve the CNN prediction accuracy
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)
#no. of training samples = 8000, no. of test samples = 2000

#Evaluation was already made during the training with the test set, therefore k-Fold Cross Validation is not needed.

#Improving the CNN
# Dropout Regularization to reduce overfitting if needed
#Dropout disables some neurons randomly at each iterations to avoid overfitting
from keras.wrappers.scikit_learn import KerasClassifier
import keras
from keras.models import Sequential     #Initialize the CNN
from keras.layers import Dense
from keras.layers import Dropout

def build_classifier():
    classifier = Sequential()
    classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
    classifier.add(Dropout(p = 0.1))        #adding Dropout to 1st hidden layer, 10% of neurons will be disabled
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    classifier.add(Dropout(p = 0.1))        #adding Dropout to 2st hidden layer, 10% of neurons will be disabled
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(units = 128, activation = 'relu'))        #hidden layer
    classifier.add(Dense(units = 1, activation = 'sigmoid'))       #output layer
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

#Below step may not work (I haven't run this code)
classifier = KerasClassifier(build_fn = build_classifier,
                             steps_per_epoch = 8000,
                             epochs = 25,
                             validation_data = test_set,
                             validation_steps = 2000)

#Parameter Tuning the ANN using Grid Search
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import keras
from keras.models import Sequential     #Initialize the ANN
from keras.layers import Dense

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(units = 128, activation = 'relu'))        #hidden layer
    classifier.add(Dense(units = 1, activation = 'sigmoid'))       #output layer
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

#Below step may not work (I haven't run this code)
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'epochs': [50,100],
              'optimizer': ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy')
grid_search = grid_search.fit(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
    