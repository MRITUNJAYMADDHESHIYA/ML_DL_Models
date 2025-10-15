#1.Need large data
#2.Types:-FNN(feedforward neural network)->input->output
#        -CNN(convolution NN)->          Grid-like data
#        -RNN(Recurrent NN)->            Sequential data(time series and natural language)
#        -GAN(Generative Adversarial N)->Generator and Discriminator(new data created)
#        -Autoencoders->                 Dimensionality reduction + noise removal + compress
#        -Transformer->                  Self-attention mechanisms(sequential data)


#Backpropagation:->After forward propagation, the network evaluates its performance using a loss function which measures the difference between the actual output and the predicted output. The goal of training is to minimize this loss. This is where backpropagation comes into play
#     loss calculation:-      using MSE or Cross-entropy loss
#     Gradient Calculation:-  The network computes the gradients of the loss function with respect to each weight and bias in the network. This involves applying the chain rule of calculus to find out how much each part of the output error can be attributed to each weight and bias.
#     Weight Update:-         Once the gradients are calculated, the weights and biases are updated using an optimization algorithm like stochastic gradient descent (SGD). The weights are adjusted in the opposite direction of the gradient to minimize the loss. The size of the step taken in each update is determined by the learning rate.

## Forward propagation -> loss calculation -> back propagation-> weight update


