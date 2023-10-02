import time

import numpy as np

import utils


class NeuralNetwork():
    def __init__(self, layer_shapes, epochs=50, learning_rate=0.01, random_state=1): #Define the constructor of the class.
        
        #Define learning paradigms
        self.epochs = epochs #Number of epochs to train the network.
        self.learning_rate = learning_rate #Learning rate for the gradient descent.
        self.random_state = random_state #Random state for the random number generator.

        #Define network architecture: no. of layers and neurons
        #layer_shapes[i] is the shape of the input that gets multiplied 
        #to the weights for the layer (e.g. layer_shapes[0] is 
        #the number of input features)
        
        self.layer_shapes = layer_shapes #The sizes of the layers of the network.
        self.weights = self._initialize_weights() #Initialize the weights of the network.
        
        #Initialize weight vectors calling the function
        #Initialize list of layer inputs before and after  
        #activation as lists of zeros.
        self.A = [None] * len(layer_shapes) #Initialize the list of layer inputs before activation as a list of zeros.
        self.Z = [None] * (len(layer_shapes)-1) #Initialize the list of layer inputs after activation as a list of zeros.

        #Define activation functions for the different layers
        self.activation_func = utils.sigmoid #The activation function of the network.
        self.activation_func_deriv = utils.sigmoid_deriv #The derivative of the activation function of the network.
        self.output_func = utils.softmax #The output function of the network.
        self.output_func_deriv = utils.softmax_deriv #The derivative of the output function of the network.
        self.cost_func = utils.mse #The cost function of the network.
        self.cost_func_deriv = utils.mse_deriv #The derivative of the cost function of the network.




    def _initialize_weights(self): #Define the method to initialize the weights of the network. 

        np.random.seed(self.random_state) #Set the random seed for the random number generator.
        self.weights = [] 

        for i in range(1, len(self.layer_shapes)):
            weight = np.random.rand(self.layer_shapes[i], self.layer_shapes[i-1]) - 0.5
            self.weights.append(weight)

        return self.weights


    def _forward_pass(self, x_train):
        '''
        TODO: Implement the forward propagation algorithm.
        The method should return the output of the network.
        '''

        #Initialize the input of the first layer as the input of the network.
        self.A[0] = x_train

        #Iterate over all layers except the last one because we need to apply the activation function to the output of the last layer.
        for i in range(len(self.layer_shapes)-1):
            #Compute the input of the layer.
            self.Z[i] = np.dot(self.weights[i], self.A[i])
            #Compute the output of the layer.
            self.A[i+1] = self.activation_func(self.Z[i])
        return self.A[-1] #Return the output of the last layer without applying the activation function.



    def _backward_pass(self, y_train, output):
        '''
        TODO: Implement the backpropagation algorithm responsible for updating the weights of the neural network.
        The method should return a list of the weight gradients which are used to update the weights in self._update_weights().

        '''

        weight_gradients = [] * len(self.weights) #Initialize the list of weight gradients as a list of zeros.
        #Compute the error of the output layer
        error_output = self.cost_func_deriv(y_train, output) * self.output_func_deriv(self.Z[-1])
        #Compute the gradient of the weights of the output layer
        weight_gradients.append(np.outer(error_output, self.A[-2]))
        #Compute the error of the hidden layers
        error_hidden = error_output
        for i in range(len(self.layer_shapes)-2, 0, -1):
            error_hidden = np.dot(self.weights[i].T, error_hidden) * self.activation_func_deriv(self.Z[i-1])
            weight_gradients.append(np.outer(error_hidden, self.A[i-1]))
        #Reverse the list of weight gradients
        weight_gradients.reverse()
        return weight_gradients

    


    def _update_weights(self,weight_gradients):
        '''
        TODO: Update the network weights according to stochastic gradient descent.

        '''
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * weight_gradients[i]
    




    def _print_learning_progress(self, start_time, iteration, x_train, y_train, x_val, y_val): #Print the learning progress of the network.
        train_accuracy = self.compute_accuracy(x_train, y_train) #Compute the training accuracy of the network.
        val_accuracy = self.compute_accuracy(x_val, y_val) #Compute the validation accuracy of the network.
        print(
            f'Epoch: {iteration + 1}, ' \
            f'Training Time: {time.time() - start_time:.2f}s, ' \
            f'Training Accuracy: {train_accuracy * 100:.2f}%, ' \
            f'Validation Accuracy: {val_accuracy * 100:.2f}%'
            )
        
        return train_accuracy, val_accuracy


    def compute_accuracy(self, x_val, y_val):
        predictions = []

        for x, y in zip(x_val, y_val):
            pred = self.predict(x)
            predictions.append(pred == np.argmax(y))

        return np.mean(predictions)


    def predict(self, x):
        '''
        TODO: Implement the prediction making of the network.
        The method should return the index of the most likeliest output class.
        '''
        return np.argmax(self._forward_pass(x))



    def fit(self, x_train, y_train, x_val, y_val): #Fit the network to the training data.

        history = {'accuracy': [], 'val_accuracy': []} #Initialize the history of the network.
        start_time = time.time()

        for iteration in range(self.epochs):
            for x, y in zip(x_train, y_train):
                output = self._forward_pass(x)
                weight_gradients = self._backward_pass(y, output)
                self._update_weights(weight_gradients)

            train_accuracy, val_accuracy = self._print_learning_progress(start_time, iteration, x_train, y_train, x_val, y_val)
            history['accuracy'].append(train_accuracy)
            history['val_accuracy'].append(val_accuracy)
        return history
