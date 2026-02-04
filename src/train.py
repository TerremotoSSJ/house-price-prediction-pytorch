import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
"""
Train a neural network model for regression tasks on housing data.
Includes early stopping based on validation loss.
"""

def train_model(model, dataloader_train,dataloader_validation, criterion=nn.MSELoss(), optimizer=None,patience=5, epochs=100, device=None):
    """
    Docstring for train_model
    
    :param model: The neural network model to be trained
    :param dataloader_train: the training data loader
    :param dataloader_validation: the validation data loader
    :param criterion: the loss function
    :param optimizer: the optimization algorithm
    :param patience: the number of epochs to wait for improvement before early stopping
    :param epochs: number of training epochs
    :param device: the device to run the training on (cpu or cuda)
    :return: the trained model, training loss history, validation loss history
    """
    #Set default optimizer and device if not provided
    if optimizer is None:
        optimizer=optim.Adam(model.parameters(), lr=0.001) #default optimizer
    if device is None:
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu") #default device
    #Begin training loop
    model.train()
    best_validation_loss=float('inf')
    loss_train_history=[] #plotting loss
    loss_validation_history=[] #plotting loss
    best_model_wts=None #best model weights
    no_improve_epochs=0 #number of epochs without improvement
    for epoch in range(epochs):
        n_seen_train=0 #number of seen samples
        total_loss_train=0 #total training loss
        total_loss_validation=0 #total validation loss
        for batch in dataloader_train: #training loop
            features,labels=batch
            features=features.to(device)
            labels=labels.to(device)
            optimizer.zero_grad()
            outputs=model(features)
            loss=criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            n_seen_train+=labels.size(0) #update number of seen samples
            total_loss_train+=loss.item()*labels.size(0) #update total training loss
        avg_loss_train=total_loss_train/n_seen_train
        loss_train_history.append(avg_loss_train)
        model.eval()
        n_seen_val=0
        with torch.no_grad(): #validation loop
            for batch in dataloader_validation: #validation loop
                features,labels=batch
                features=features.to(device)
                labels=labels.to(device)
                outputs=model(features)
                loss=criterion(outputs, labels)
                n_seen_val+=labels.size(0)
                total_loss_validation+=loss.item()*labels.size(0)
        avg_loss_validation=total_loss_validation/n_seen_val
        loss_validation_history.append(avg_loss_validation)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss_train}, Validation Loss: {avg_loss_validation}")
        if(avg_loss_validation<best_validation_loss): #early stopping logic
            no_improve_epochs=0
            best_validation_loss=avg_loss_validation
            best_model_wts=model.state_dict() #save best model weights
        else:
            no_improve_epochs+=1
            if(no_improve_epochs>=patience): #early stopping
                print("Early stopping")
                break
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts) #load best model weights
    else: 
        print("No improvement during training") 
    model = model.to(device) #ensure model is on the correct device
    return model, loss_train_history, loss_validation_history #return trained model and loss history      
