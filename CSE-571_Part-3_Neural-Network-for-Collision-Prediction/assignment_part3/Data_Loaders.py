import torch
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader


class Nav_Dataset(dataset.Dataset):
    def __init__(self):
        self.data = np.genfromtxt('saved/training_data.csv', delimiter=',')
        # STUDENTS: it may be helpful for the final part to balance the distribution of your collected data
        labels = self.data[:, -1]
        zeros = np.sum(labels == 0)
        ones = np.sum(labels == 1)
        if zeros > ones:
            self.data = self.data[labels == 0][:ones]
        elif ones > zeros:
            self.data = self.data[labels == 1][:zeros]

        self.lenDataset = len(self.data)

        # normalize data and save scaler for inference
        self.scaler = MinMaxScaler()
        self.normalized_data = self.scaler.fit_transform(self.data) #fits and transforms
        pickle.dump(self.scaler, open("saved/scaler.pkl", "wb")) #save to normalize at inference

    def __len__(self):
    # STUDENTS: __len__() returns the length of the dataset
        return self.lenDataset

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()
        # STUDENTS: for this example, __getitem__() must return a dict with entries {'input': x, 'label': y}
        # x and y should both be of type float32. There are many other ways to do this, but to work with autograding
        # please do not deviate from these specifications.

        sample = self.normalized_data[idx, :-1]
        label = self.normalized_data[idx, -1]

        return {'input': torch.tensor(sample, dtype = torch.float32),
                'label': torch.tensor(label, dtype = torch.float32)}


class Data_Loaders():
    def __init__(self, batch_size):
        self.nav_dataset = Nav_Dataset()
        # STUDENTS: randomly split dataset into two data.DataLoaders, self.train_loader and self.test_loader
        # make sure your split can handle an arbitrary number of samples in the dataset as this may vary

        dataset_len = len(self.nav_dataset)
        trainData_len = int(0.8 * dataset_len)
        testData_len = dataset_len - trainData_len
        self.train_dataset, self.test_dataset = dataset.random_split(self.nav_dataset, [trainData_len, testData_len])

        self.batch_size = batch_size
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def train_loader(self):
        return self.train_loader
    def test_loader(self):
        return self.test_loader

def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)

    # STUDENTS : note this is how the dataloaders will be iterated over, and cannot be deviated from
    for idx, sample in enumerate(data_loaders.train_loader):
        _, _ = sample['input'], sample['label']
    for idx, sample in enumerate(data_loaders.test_loader):
        _, _ = sample['input'], sample['label']

if __name__ == '__main__':
    main()
