import torch
import torch.nn as nn
"""
Create a model for regression tasks on housing data.
The model consists of multiple linear layers with ReLU activations and dropout for regularization.
"""
class HousingModel(nn.Module):
    def __init__(self, input_size, base_neurons):  #input size is number of features in dataset
        """
        Docstring for __init__
        
        :param input_size: number of input features
        :param base_neurons: number of neurons in the first hidden layer
        """
        
        super().__init__()
        self.model=nn.Sequential( #model definition
            nn.Linear(input_size, base_neurons), #input layer
            nn.ReLU(),
            nn.Dropout(0.2), #dropout for regularization
            nn.Linear(base_neurons, base_neurons//2), #hidden layer
            nn.ReLU(), 
            nn.Dropout(0.2), #dropout for regularization
            nn.Linear(base_neurons//2, 1) #reggression output only 1 value
        )
    def forward(self, x):
        return self.model(x)
    