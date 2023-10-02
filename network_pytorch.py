import time

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np



#We will use the following class to implement the neural network in PyTorch.
class NeuralNetworkTorch(nn.Module):
    #The constructor of the class takes the following parameters:
    def __init__(self, sizes, epochs=10, learning_rate=0.01, random_state=1):
        
        '''
        TODO: Implement the forward propagation algorithm.
        The layers should be initialized according to the sizes variable.
        The layers should be implemented using variable size analogously to
        the implementation in network_pytorch: sizes[i] is the shape 
        of the input that gets multiplied to the weights for the layer.
        '''
        
        super().__init__() #This line is needed to initialize the nn.Module properly.

        self.epochs = epochs #Number of epochs to train the network.
        self.learning_rate = learning_rate #Learning rate for the gradient descent.
        self.random_state = random_state   #Random state for the random number generator.
        torch.manual_seed(self.random_state) #Set the random seed for the random number generator.

        self.layers = nn.ModuleList()#This list will contain the layers of the network.
        for i in range(len(sizes) - 1):#Iterate over the sizes of the network.
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))#Add a new layer to the network.

        
        self.activation_func = torch.sigmoid #The activation function of the network.

        self.loss_func = nn.CrossEntropyLoss() #The loss function of the network. There is a difference with BCELoss since CrossEntropy is for multiclass classification that here will help us to avoid the one-hot encoding of the labels and is more appropriate for the task.
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate) #The optimizer of the network.

        #The following line is needed to initialize the weights of the network.
    def _forward_pass(self, x_train):
        '''
        TODO: The method should return the output of the network.
        '''

        for layer in self.layers[:-1]:#Iterate over all layers except the last one because we need to apply the activation function to the output of the last layer.
            x_train = self.activation_func(layer(x_train))#Apply the activation function to the output of the layer.

        return self.layers[-1](x_train) #Return the output of the last layer without applying the activation function.



    def _backward_pass(self, y_train, output):
        '''
        TODO: Implement the backpropagation algorithm responsible for updating the weights of the neural network.

        '''
        self.optimizer.zero_grad() #Reset the gradients of the network.
        loss = self.loss_func(output, y_train) #Compute the loss of the network.
        loss.backward() #Compute the gradients of the network.




    def _update_weights(self):#Update the network weights according to stochastic gradient descent.
        '''
        TODO: Update the network weights according to stochastic gradient descent.
        '''
        self.optimizer.step()#Update the weights of the network.


    def _flatten(self, x): #Flatten the input x.
        return x.view(x.size(0), -1)       #Flatten the input x.


    def _print_learning_progress(self, start_time, iteration, train_loader, val_loader):#Print the learning progress of the network.
        train_accuracy = self.compute_accuracy(train_loader)
        val_accuracy = self.compute_accuracy(val_loader)
        print(
            f'Epoch: {iteration + 1}, ' \
            f'Training Time: {time.time() - start_time:.2f}s, ' \
            f'Learning Rate: {self.optimizer.param_groups[0]["lr"]}, ' \
            f'Training Accuracy: {train_accuracy * 100:.2f}%, ' \
            f'Validation Accuracy: {val_accuracy * 100:.2f}%'
            )
        return train_accuracy, val_accuracy


    def predict(self, x):
        '''
        TODO: Implement the prediction making of the network.
        The method should return the index of the most likeliest output class.
        '''
        x = self._flatten(x) #Flatten the input x.
        output = self._forward_pass(x) #Compute the output of the network.
        return torch.argmax(output, dim=1) #Return the index of the most likeliest output class.



    def fit(self, train_loader, val_loader): #Fit the network to the training data.
        start_time = time.time() #Get the current time.
        history = {'accuracy': [], 'val_accuracy': []} #Initialize the history of the network.

        for iteration in range(self.epochs): #Iterate over the epochs of the network.
            for x, y in train_loader: #Iterate over the training data.
                x = self._flatten(x) #Flatten the input x.
                self.optimizer.zero_grad() #Reset the gradients of the network.


                output = self._forward_pass(x) #Compute the output of the network. 
                self._backward_pass(y, output) #Compute the gradients of the network.
                self._update_weights() #Update the weights of the network.

            train_accuracy, val_accuracy = self._print_learning_progress(start_time, iteration, train_loader, val_loader)
            history['accuracy'].append(train_accuracy)
            history['val_accuracy'].append(val_accuracy)

        return history



    def compute_accuracy(self, data_loader):
        correct = 0
        for x, y in data_loader:
            pred = self.predict(x)
            correct += torch.sum(torch.eq(pred, y))

        return correct / len(data_loader.dataset)
