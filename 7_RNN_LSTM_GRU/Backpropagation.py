import numpy as np
from numpy.linalg import norm
import copy
import os

def relu(x):  #Adds non-linearity and removes negative values, helping neural networks model complex patterns.
    return x*(np.sign(x)+1.)/2.
def sigmoid(x): #any real number to the range (0, 1)
    return 1./(1.+np.exp(-x))
def softmax(x): #converts a vector of values into a probability distribution
    return np.exp(x)/sum(np.exp(x))
def mynorm(z): #Root Mean Square (RMS)
    return np.sqrt(np.mean(z**2))

#Y->training output
#Xtrain->training input
#Xpred->input for prediction

def myANN(Y, Xtrain, Xpred, w01, w02, w03, b01, b02, b03):
    #Initialization
    w1 = copy.copy(w01)
    w2 = copy.copy(w02)
    w3 = copy.copy(w03)
    b1 = copy.copy(b01)
    b2 = copy.copy(b02)
    b3 = copy.copy(b03)

    k=1
    change = 999

    #training loop
    while(change>0.001 and k<201):
        print('Iteration:', k)

        #start feedforward
        z1   = sigmoid(w1 @ Xtrain + b1)  #hidden layer 1
        z2   = sigmoid(w2 @ z1 + b2)      #hidden layer 2
        Yhat = softmax(w3@ z2 + b3)       #output layer
        loss = -Y @ np.log(Yhat)          #cross-entropy loss
        print("current loss:", loss)

        ##find gradient of loss w.r.t. each weight
        #output layer
        dLdb3 = Yhat - Y 
        dLdW3 = np.outer(dLdb3, z2)
        # Hidden Layer 2
        dLdb2 = (w3.T @ (dLdb3)) * z2 * (1-z2)
        dLdW2 = np.outer(dLdb2,z1)
        # Hidden Layer 1
        dLdb1 = (w2.T @ (dLdb2)) * z1 * (1-z1)
        dLdW1 = np.outer(dLdb1, Xtrain)
        
        ## Update Weights by Back Propagation
        # Output Layer
        b3 -= dLdb3 # (learning rate is one)
        w3 -= dLdW3
        # Hidden Layer 2
        b2 -= dLdb2
        w2 -= dLdW2
        # Hidden Layer 1
        b1 -= dLdb1
        w1 -= dLdW1

        change = norm(dLdb1)+norm(dLdb2)+norm(dLdb3)+norm(dLdW1)+norm(dLdW2)+norm(dLdW3)
        k += 1
        
    Z1pred = w1 @ Xpred + b1
    Z2pred = w2 @ sigmoid(Z1pred) + b2
    Z3pred = w3 @ sigmoid(Z2pred) + b3
    Ypred = softmax(Z3pred)
    print("")
    print("Summary")
    print("Target Y \n", Y)
    print("Fitted Ytrain \n", Yhat)
    print("Xpred\n", Xpred)
    print("Fitted Ypred \n", Ypred)
    print("Weight Matrix 1 \n", w1)
    print("Bias Vector 1 \n", b1)
    print("Weight Matrix 2 \n", w2)
    print("Bias Vector 2 \n", b2)
    print("Weight Matrix 3 \n", w3)
    print("Bias Vector 3 \n", b3)


## Initial weights and biases
W0_1 = np.array([[0.1,0.3,0.7], [0.9,0.4,0.4]])
b_1 = np.array([1.,1.])

W0_2 = np.array([[0.4,0.3], [0.7,0.2]])
b_2 = np.array([1.,1.])

W0_3 = np.array([[0.5,0.6], [0.6,0.7], [0.3,0.2]])
b_3 = np.array([1.,1.,1.]) 

#training data
X_train = np.array([0.1, 0.7, 0.3])
YY      = np.array([1., 0., 0.])
X_pred  = X_train

###myANN(YY, X_train, X_pred, W0_1, W0_2, W0_3, b_1, b_2, b_3)


##########################################################
#Keras provides a rich suite of model architectures, layers, activation functions and other building blocks for creating deep learning models.
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.initializers import Constant

# Initial weights and biases
W0_1 = np.array([[0.1,0.3,0.7], [0.9,0.4,0.4]])  # shape (2,3)
b_1  = np.array([1.,1.])                         # shape (2,)

W0_2 = np.array([[0.4,0.3], [0.7,0.2]])          # shape (2,2)
b_2  = np.array([1.,1.])                         # shape (2,)

W0_3 = np.array([[0.5,0.6], [0.6,0.7], [0.3,0.2]]) # shape (3,2)
b_3  = np.array([1.,1.,1.])                        # shape (3,)

# Training data
X_train = np.array([[0.1, 0.7, 0.3]])  # shape (1,3)
YY      = np.array([[1., 0., 0.]])     # shape (1,3)

# Create the model
model = Sequential()
model.add(Dense(2, input_dim=3, activation='sigmoid'))  # first hidden layer
model.add(Dense(2, activation='sigmoid'))               # second hidden layer
model.add(Dense(3, activation='softmax'))               # output layer

# Build the model (needed before setting weights)
model.build(input_shape=(None, 3))

# Set initial weights for each layer
model.layers[0].set_weights([W0_1.T, b_1])
model.layers[1].set_weights([W0_2.T, b_2])
model.layers[2].set_weights([W0_3.T, b_3])

# Compile the model
sgd = SGD(learning_rate=1.0)  # learning rate = 1
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['categorical_crossentropy'])

# Train the model
model.fit(X_train, YY, epochs=200, batch_size=1)

# Check final weights
print("Trained weights:")
for i, layer in enumerate(model.layers):
    w, b = layer.get_weights()
    print(f"Layer {i+1} weights:\n{w}")
    print(f"Layer {i+1} biases:\n{b}")

model.predict(X_pred.reshape((1, 3)))
model.get_weights()