import torch
from torchvision import datasets, transforms
import numpy as np
from numpy import load, concatenate

class CustomDataSet(torch.utils.data.Dataset):
    def __init__(self,path='data/processed/train.pt'):
        data = torch.load(path)
        print(data['labels'])
        self.images = data['images']
        self.labels = data['labels']
        #self.images = self.images.astype(np.float)
        #self.labels = self.labels.astype(np.float)
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.images[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

def get_train_loader():
    train = CustomDataSet()
    trainloader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
    return trainloader

def get_test_loader(path='data/processed/test.pt'):
    test = CustomDataSet()
    testloader = torch.utils.data.DataLoader(test, batch_size=64, shuffle=True)
    return testloader
