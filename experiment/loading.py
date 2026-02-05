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

name=input("Name of the model to load (without extension): ") #ask user for model name
model=HousingModel.load(name) #load model
dataset=housingDataset() #create dataset instance
features, labels = dataset.random_values() #get random sample from dataset
predicted=model(features) #test forward pass with random
print(f"Predicted: {predicted.item():.6f}, Actual: {labels.item():.6f}") #print predicted and actual values