import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


def create_dataset(dataset, seq_length=1, device="cuda:0"):
    dataX, dataY = [], []
    for i in range(len(dataset)-seq_length-1):
        dataX.append(torch.from_numpy(dataset[i:(i+seq_length), 0]).unsqueeze(0))
        dataY.append(torch.from_numpy(np.array([dataset[i + seq_length, 0]])).unsqueeze(0))
        
    dataX = torch.cat(dataX) if len(dataX) > 0 else torch.tensor([])
    dataY = torch.cat(dataY) if len(dataY) > 0 else torch.tensor([])
    return dataX.to(device), dataY.to(device)


def get_data_loaders(trainX, trainY, batch_size=32):
    
    train_loader = DataLoader(TensorDataset(trainX, trainY), batch_size=batch_size, shuffle=True)
        
    return train_loader