import torch
from torch.utils.data import DataLoader, Subset
import pickle
import os


def save_dataset_split(dataset, train_indices, val_indices, filename):
    """Save dataset split indices to a file"""
    split_data = {
        'train_indices': train_indices,
        'val_indices': val_indices
    }

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(split_data, f)


def load_dataset_split(dataset, filename, batch_size=32):
    """Load dataset split indices and create corresponding DataLoader"""
    with open(filename, 'rb') as f:
        split_data = pickle.load(f)

    # Create validation subset
    val_subset = Subset(dataset, split_data['val_indices'])

    # Create and return validation loader
    return DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )