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
model.eval()  # ¡IMPORTANTE!

# Y asegúrate de usar torch.no_grad():
with torch.no_grad():
    dataset=housingDataset() #create dataset instance
    try:
        number=int(input("Number of random samples to test: (Default=1)")) #ask user for number of samples
    except ValueError:
        number=1 #default to 1 sample if invalid input

    average_loss=0.0
    criterion=torch.nn.MSELoss() #define loss function
    for i in range(number):
        features, labels = dataset.random_values() #get random sample from dataset
        predicted=model(features) #test forward pass with random
        loss=criterion(predicted, labels)
        average_loss+=loss.item() #accumulate loss
        print(f"Sample {i+1}: Predicted={predicted.item():.6f}, Actual={labels.item():.6f}, Loss={loss.item():.6f}") #print loss for each sample

    print(f"Average Loss: {average_loss/number:.6f}") #print average loss