import torch
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class Nav_Dataset(dataset.Dataset):
    def __init__(self):
        self.data = np.genfromtxt('saved/training_data.csv', delimiter=',')

        print(f"Data before pruning: {self.data.shape}")

        self.data = self.filter_data(self.data)
        print("filtered: ", len(self.data))

        labels = self.data[:, -1]

        # Finding indices of samples with label zero and non-zero
        zero_indices = np.where(labels == 0)[0]
        non_zero_indices = np.where(labels != 0)[0]
        print("zi length: ", len(zero_indices))
        print("nzi length: ", len(non_zero_indices))

        # Calculate the desired number of samples for balancing
        min_zero_ratio = int(0.50 * len(zero_indices))
        min_non_zero_ratio = int(0.50 * len(non_zero_indices))

        # Ensure that the number of selected non-zero samples is at most max_zero_samples
        selected_non_zero_indices = np.random.choice(non_zero_indices, min_non_zero_ratio,
                                                     replace=False)

        selected_zero_indices = np.random.choice(zero_indices, min_zero_ratio,
                                                     replace=False)

        print("selected_non_zero_indices length: ", len(selected_non_zero_indices))
        print("selected_zero_indices length: ", len(selected_zero_indices))

        # Combine the selected samples
        selected_indices = np.concatenate([selected_zero_indices, selected_non_zero_indices])

        # Use the selected indices to create the pruned dataset
        self.data = self.data[selected_indices]

        print(f"Data after pruning: {self.data.shape}")

        # Normalize data and save scaler for inference
        self.scaler = MinMaxScaler()
        self.normalized_data = self.scaler.fit_transform(self.data)
        pickle.dump(self.scaler, open("saved/scaler.pkl", "wb"))

        print(f"Normalized data count: {len(self.normalized_data)}")

    def filter_data(self, data):
        sensor_values = data[:, :-2]  # Exclude the action and collision columns
        collision_labels = data[:, -1]

        # Find rows where all sensor values are 150 and collision = 1
        filter_condition = np.all(sensor_values == 150, axis=1) & (collision_labels == 1)

        # Keep rows that don't meet the filter condition
        filtered_data = data[~filter_condition]
        return filtered_data

    def __len__(self):
        return len(self.normalized_data)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()
        sample = {'input': torch.tensor(self.normalized_data[idx, :-1], dtype=torch.float32),
                  'label': torch.tensor(self.normalized_data[idx, -1], dtype=torch.float32)}
        return sample


class Data_Loaders():
    def __init__(self, batch_size, test_size=0.2, random_seed=42):
        self.nav_dataset = Nav_Dataset()

        # Randomly split dataset into two data.DataLoaders
        train_data, test_data = train_test_split(self.nav_dataset, test_size=test_size, random_state=random_seed)
        self.train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.test_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=False)


def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)

    print(f"Training dataset length: {len(data_loaders.train_loader.dataset)}")
    print(f"Testing dataset length: {len(data_loaders.test_loader.dataset)}")
    # Note: This is how the dataloaders will be iterated over and cannot be deviated from
    for idx, sample in enumerate(data_loaders.train_loader):
        _, _ = sample['input'], sample['label']

    for idx, sample in enumerate(data_loaders.test_loader):
        _, _ = sample['input'], sample['label']


if __name__ == '__main__':
    main()



