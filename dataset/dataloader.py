from torch.utils.data import Dataset, DataLoader, random_split
from .data_prep import TennisTrajectoryDataset

def create_dataloader(file_path, batch_size=64, seq_length=20):
    full_dataset = TennisTrajectoryDataset(file_path, seq_length)

    # Split into train (70%), val (15%), test (15%)
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, val_dataloader,test_dataloader #DataLoader(dataset, batch_size=batch_size, shuffle=True)