
import sys
import os
import torch

# Adjust the import path to include the 'src' directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get parent directory
parent_dir = os.path.dirname(current_dir)
# Insert 'src' directory to sys.path
sys.path.insert(0, os.path.join(parent_dir, 'src'))

from dataset import housingDataset  
from model import HousingModel
from train import train_model
from eval import evaluation_model
from torch.utils.data import DataLoader, random_split
"""Main script to train and evaluate the HousingModel on housing data."""


device=torch.device("cuda" if torch.cuda.is_available() else "cpu") #set device
#First create dataset
dataset=housingDataset() #create dataset instance
#Split dataset into training and validation sets
train_size=int(0.7*len(dataset)) #70% for training
val_size=int(0.1*len(dataset)) #10% for validation
test_size=len(dataset)-train_size-val_size #20% for testing
dataset_train, dataset_val, dataset_test=random_split(dataset, [train_size, val_size, test_size]) #split dataset
#Create data loaders
dataloader_train=DataLoader(dataset_train, batch_size=32, shuffle=True) #training data loader
dataloader_val=DataLoader(dataset_val, batch_size=32, shuffle=False) #validation data loader
dataloader_test=DataLoader(dataset_test, batch_size=32, shuffle=False) #test data loader
#Create model instance
input_size=dataset.features.shape[1] #number of features
model=HousingModel(input_size=input_size,base_neurons=64) #create model
model=model.to(device) #move model to device
#Train model
optimizer=torch.optim.Adam(model.parameters(), lr=0.001) #define optimizer
criterion=torch.nn.MSELoss() #define loss function
model, train_loss_history, val_loss_history=train_model(model, dataloader_train, dataloader_val, criterion=criterion, optimizer=optimizer, device=device, epochs=100) #train model
#Evaluate model on test set
test_loss=evaluation_model(model, dataloader_test, device=device) #evaluate model
print(f"Test Loss: {test_loss}")