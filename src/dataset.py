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
Regularization will be done in the model with dropout layers
"""

"""
There was an error in the previous version where i didnt handle missing values in the dataset since
i thought there were none, but there are missing values in the total_bedrooms column.
We will handle them by imputing with the median of the column.
"""
class housingDataset(Dataset):
    def __init__(self):
        super().__init__()
        data=pd.read_csv("githubpro/data/housing.csv") #read csv
        #handle missing values by imputing with median
        median_bedrooms = data['total_bedrooms'].median()
        data['total_bedrooms'].fillna(median_bedrooms, inplace=True)
        #first we have to convert categorical data to numerical data
        data=pd.get_dummies(data, columns=['ocean_proximity']) #one hot encoding
        features_raw=data.drop('median_house_value', axis=1).values
        labels_raw=data['median_house_value'].values
        self.scalerFeatures=StandardScaler()
        features_norm=self.scalerFeatures.fit_transform(features_raw) #standardization
        self.scalerLabels=StandardScaler()
        labels_norm=self.scalerLabels.fit_transform(labels_raw.reshape(-1,1)).flatten() #standardization
        self.features=torch.tensor(features_norm, dtype=torch.float32)
        self.labels=torch.tensor(labels_norm, dtype=torch.float32).view(-1,1) #view to make it a column vector
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx] #return features and labels as a tuple for efficiency
    def random_values(self):
        idx=torch.randint(0, len(self.features), (1,)).item() #random index
        return self.features[idx], self.labels[idx] #return features and labels as a tuple for efficiency


        