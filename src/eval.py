
import torch
import torch.nn as nn
"""
Evaluate a neural network model for regression tasks on housing data.
Calculates average loss over the evaluation dataset.
"""
def evaluation_model(model, dataloader, criterion=nn.MSELoss(), device=None):
    """
    Evaluate the model on the given dataloader.
    Args:
        model: Trained neural network model.
        dataloader: DataLoader for evaluation data.
        criterion: Loss function.
        device: Device to run the evaluation on.
    Returns:
        avg_loss: Average loss over the evaluation dataset.
    """
    #Set default device if not provided
    if device is None:
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu") #default device
    
    #Begin evaluation loop
    model = model.to(device)
    model.eval() #set model to evaluation mode
    total_loss=0.0
    n_seen=0
    with torch.no_grad():
        for batch in dataloader:
            features,labels=batch
            features=features.to(device)
            labels=labels.to(device)
            outputs=model(features)
            loss=criterion(outputs, labels)
            batch_size=features.size(0)
            total_loss+=loss.item()*batch_size #accumulate loss
            n_seen+=batch_size #accumulate number of samples seen
    avg_loss=total_loss/n_seen #compute average loss over all samples
    return avg_loss