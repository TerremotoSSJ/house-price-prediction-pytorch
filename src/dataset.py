import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
"""
First of all we create the dataset, we have to read the csv and separate features and labels
for features there is a variable that is not numerical, so we have to convert them to numerical
we use one hot encoding for that as the first approach
Also we need to do standardization because the values are in different scales and gradient
would vary a lot between features
"""
class housingDataset(Dataset):
    def __init__(self):
        super().__init__()
        data=pd.read_csv("githubpro/data/housing.csv") #read csv
        #first we have to convert categorical data to numerical data
        data=pd.get_dummies(data, columns=['ocean_proximity']) #one hot encoding
        self.features=data.drop('median_house_value', axis=1).values
        self.labels=data['median_house_value'].values
        self.scalerFeatures=StandardScaler()
        self.features=self.scalerFeatures.fit_transform(self.features) #standardization
        self.scalerLabels=StandardScaler()
        self.labels=self.scalerLabels.fit_transform(self.labels.reshape(-1,1)).flatten() #standardization
        self.features=torch.tensor(self.features, dtype=torch.float32)
        self.labels=torch.tensor(self.labels, dtype=torch.float32).view(-1,1) #view to make it a column vector
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx] #return features and labels as a tuple for efficiency


        