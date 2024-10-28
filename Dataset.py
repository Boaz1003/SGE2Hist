import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch



class DataSet(Dataset):
    def __init__(self, datadir, transform=None):
        self.datadir = datadir
        self.transform = transform
        with open(self.datadir, 'rb') as f:
            self.dataset = pickle.load(f)
        # keys = list(self.dataset.keys())
        # train_keys, test_keys = train_test_split(keys, test_size=test_size, random_state=42)
        #
        # self.keys = train_keys if split == 'train' else test_keys
        self.keys = list(self.dataset.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        data = self.dataset[str(self.keys[idx])]
        image = data['image']
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image).float()
        image = image / 255.
        if self.transform:
            image = self.transform(image)
        gene = data['gene']
        # gene = gene.toarray()
        gene = torch.from_numpy(gene).float()
        gene = torch.squeeze(gene, dim=0)
        type = data['type']
        type = torch.tensor(type)
        return image, gene, type

    def split_dataset(dataset, train_ratio=0.8):
        dataset_size = len(dataset)
        train_size = int(train_ratio * dataset_size)
        test_size = dataset_size - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        return train_dataset, test_dataset
