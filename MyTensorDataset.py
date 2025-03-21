import torch
from torch.utils.data import Dataset
import os

class MyTensorDataset(Dataset):
    def __init__(self, root_dir):
        """
        Args:
            root_dir (str): The root directory containing label folders.
        """
        self.root_dir = root_dir
        self.samples = []
        self.labels = []

        # Traverse the subfolders (labels)
        for label_idx, label_folder in enumerate(os.listdir(root_dir)):
            label_path = os.path.join(root_dir, label_folder)
            
            if os.path.isdir(label_path):
                # List all .pth files in this label folder
                for file_name in os.listdir(label_path):
                    if file_name.endswith('.pth'):
                        file_path = os.path.join(label_path, file_name)
                        self.samples.append(file_path)
                        self.labels.append(label_idx)  # Store the label index

    def __len__(self):
        # The number of samples (i.e., the number of .pth files)
        return len(self.samples)

    def __getitem__(self, idx):
        # Get the file path and load the tensor
        file_path = self.samples[idx]
        tensor = torch.load(file_path, weights_only=True)
        label = self.labels[idx]
        return tensor, label

