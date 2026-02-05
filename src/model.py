import os
import torch
import torch.nn as nn
"""
Create a model for regression tasks on housing data.
The model consists of multiple linear layers with ReLU activations and dropout for regularization.
"""
class HousingModel(nn.Module):
    def __init__(self, input_size, base_neurons=64):  #input size is number of features in dataset
        """
        Docstring for __init__
        
        :param input_size: number of input features
        :param base_neurons: number of neurons in the first hidden layer
        """
        
        super().__init__()
        self.input_size = input_size
        self.base_neurons = base_neurons
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
    def save(self, name):
        """
        Save the model to a path based on the given name.
        """
        model_path = f"models/{name}.pth" #save model in models directory with .pth extension
        os.makedirs("models", exist_ok=True)  # Create directory if it doesn't exist
        torch.save({ #Save the hipperparameters and state dict
            'model_state_dict': self.state_dict(),
            'input_size': self.input_size,
            'base_neurons': self.base_neurons
        }, model_path
        ) 
    '''
    If the model doesn't have the same hyperparameters as when it was saved, loading will fail.
    That's why we use a class method to create the model with the correct hyperparameters.
    cls means the class itself, so we can create an instance of the model inside the load method.
    '''
    @classmethod #class method to load model from file
    def load(cls, name): 
        """
        Load the model from a path based on the given name.
        :param name: name of the model file (without extension)
        :return: the loaded model
        """
        model_path = f"models/{name}.pth" #model path
        checkpoint = torch.load(model_path) #load checkpoint
        model = cls(input_size=checkpoint['input_size'], base_neurons=checkpoint['base_neurons']) #create model instance
        model.load_state_dict(checkpoint['model_state_dict']) #load state dict
        return model
    