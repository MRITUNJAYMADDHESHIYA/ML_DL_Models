import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_moons

###No-hidden layers --> one hidden layers --> two hidden layes
X, Y = make_moons(n_samples=100, noise=0)
colors = np.array(['#377eb8', '#ff7f00'])

def create_logit():
    model = Sequential()
    #only layer, transforming the 2d-->1d output
    model.add(Dense(1, input_dim=2, kernel_initializer='normal', activation='sigmoid'))
    #compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def plot_decision_surface(X, model, h):
    # Get the upper and lower bounds of the two input dimensions
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    
    # Create the mesh over the input space
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    y_pred = model.predict(np.c_[xx.ravel(), yy.ravel()])
    
    # Quantise the model output to the integers 0 & 1
    g_pred = np.array(y_pred > 0.5, dtype=np.integer)
    
    # Plot the inputs and decision boundary on the input space
    plt.scatter(xx.ravel(), yy.ravel(), s=10, color=colors[g_pred.ravel()])    
    plt.scatter(X[:, 0], X[:, 1], s=30, color=colors[Y], edgecolors='black')
    plt.xlabel(r'$X_1$')
    plt.ylabel(r'$X_2$')
    return plt

seed = 56
np.random.seed(seed)
tf.random.set_seed(seed)

############ Evaluate the logit model ##########
model = create_logit()
# model.fit(X, Y, epochs=300, batch_size=1, verbose=1)



############ Pridict classes of training ppoints
y_pred = model.predict(X)
g_pred = (y_pred > 0.5)

def logit(x):
    return 1/(1+np.exp(-x))

w1, b1 = model.layers[0].get_weights()
for i in range(0, len(X)):
    x = X[i]
    y_hat = logit(np.matmul(np.transpose(w1), x) + b1)
    print(y_hat, y_pred[i])



################### Plot the separating hyperplanes #####################
x_min = min(X[:, 0])
x_max = max(X[:, 0])

y1 = -(b1 + w1[0]*x_min)/w1[1]
y2 = -(b1 + w1[0]*x_max)/w1[1]

plt.scatter(X[:, 0], X[:, 1], s=30, color=colors[Y])
plt.plot([x_min, x_max], [y1, y2], color='k', linestyle='-', linewidth=2)
plt.xlabel(r'$X_1$')
plt.ylabel(r'$X_2$');

plot_decision_surface(X, model, 0.01)