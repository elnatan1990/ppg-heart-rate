import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime
import os
from models import PPGNet, PPGDataset, NoiseClassifier, NoiseDataset
from dataset_utils import save_dataset_split

def create_dataloaders(ppg_signals, batch_size=32, val_split=0.1):
    """Create train and validation dataloaders for HR estimation"""
    from models import PPGDataset
    from torch.utils.data import Subset
    import numpy as np

    # Create full dataset
    dataset = PPGDataset(ppg_signals['signal'].values, ppg_signals['hr'].values)

    # Calculate sizes
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_split)
    indices = list(range(dataset_size))

    # Create splits using the same random seed
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[val_size:], indices[:val_size]

    # Create train and validation subsets
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    # Create dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # Save the split
    save_dataset_split(dataset, train_indices, val_indices, 'data_splits/hr_split.pkl')

    return train_loader, val_loader

def create_noise_dataloaders(ppg_signals, batch_size=32, val_split=0.1):
    """Create train and validation dataloaders for noise classification"""
    dataset = NoiseDataset(ppg_signals)

    # Calculate lengths for split
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    # Split dataset
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader

def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs, device):
    """Training loop with early stopping"""
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    patience = 5
    best_model_state = None
    best_epoch = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # Handle different label shapes for HR estimation vs noise classification
            if isinstance(criterion, nn.CrossEntropyLoss):
                loss = criterion(outputs, labels)
            else:
                loss = criterion(outputs, labels.float())

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)

                if isinstance(criterion, nn.CrossEntropyLoss):
                    loss = criterion(outputs, labels)
                    pred = torch.argmax(outputs, dim=1)
                else:
                    loss = criterion(outputs, labels.float())
                    pred = outputs

                val_loss += loss.item()
                val_predictions.extend(pred.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Calculate metrics
        if isinstance(criterion, nn.CrossEntropyLoss):
            val_acc = np.mean(np.array(val_predictions) == np.array(val_targets))
            print(f'Epoch {epoch + 1}/{num_epochs}:')
            print(f'Train Loss: {avg_train_loss:.4f}')
            print(f'Val Loss: {avg_val_loss:.4f}')
            print(f'Val Accuracy: {val_acc:.4f}')
        else:
            val_mae = np.mean(np.abs(np.array(val_predictions) - np.array(val_targets)))
            print(f'Epoch {epoch + 1}/{num_epochs}:')
            print(f'Train Loss: {avg_train_loss:.4f}')
            print(f'Val Loss: {avg_val_loss:.4f}')
            print(f'Val MAE: {val_mae:.4f} BPM')

        print('-' * 50)

        # Early stopping and model saving check
        if avg_val_loss < best_val_loss:
            print(f'Validation loss decreased from {best_val_loss:.4f} to {avg_val_loss:.4f}. Saving model...')
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0
            # Save best model state
            best_model_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }
        else:
            patience_counter += 1
            print(f'Validation loss did not decrease. Best was {best_val_loss:.4f} at epoch {best_epoch + 1}.')
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs. '
                      f'Best validation loss was {best_val_loss:.4f} at epoch {best_epoch + 1}')
                break

    # Save the best model state
    if best_model_state is not None:
        os.makedirs('models', exist_ok=True)
        if isinstance(criterion, nn.CrossEntropyLoss):
            torch.save(best_model_state, 'models/noise_model.pth')
            print(f'Saved best noise model from epoch {best_epoch + 1} with validation loss {best_val_loss:.4f}')
        else:
            torch.save(best_model_state, 'models/hr_model.pth')
            print(f'Saved best HR model from epoch {best_epoch + 1} with validation loss {best_val_loss:.4f}')

    return train_losses, val_losses

def evaluate_model(model, test_loader, device):
    """Evaluate model performance"""
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            predictions.extend(outputs.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    predictions = np.array(predictions)
    targets = np.array(targets)
    mae = np.mean(np.abs(predictions - targets))

    return mae, predictions, targets

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001 # old 0.001
    NUM_EPOCHS = 50

    # Challenge 1: Basic HR Estimation
    print("Starting Challenge 1: Basic HR Estimation")

    # Load data
    ppg_signals = pd.read_parquet('ppg_samples_with_hr.parquet')

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(ppg_signals, BATCH_SIZE)

    # Initialize model, criterion, and optimizer
    model = PPGNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train model
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        NUM_EPOCHS, device
    )

    # Evaluate final model
    model.eval()
    val_mae = 0
    n_samples = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            val_mae += torch.sum(torch.abs(outputs - labels)).item()
            n_samples += len(inputs)

    val_mae /= n_samples
    print(f'Challenge 1 Validation MAE: {val_mae:.4f} BPM')

    # Challenge 2: Noisy Signal Detection
    print("\nStarting Challenge 2: Noisy Signal Detection")

    # Load challenge 2 data
    ppg_signals_v4 = pd.read_parquet('ppg_samples_with_hr_v4.parquet')

    # Create noise dataloaders
    noise_train_loader, noise_val_loader = create_noise_dataloaders(ppg_signals_v4, BATCH_SIZE)

    # Initialize noise classifier
    noise_model = NoiseClassifier().to(device)
    noise_criterion = nn.CrossEntropyLoss()
    noise_optimizer = torch.optim.Adam(noise_model.parameters(), lr=LEARNING_RATE)

    # Train noise classifier
    noise_train_losses, noise_val_losses = train_model(
        noise_model, noise_train_loader, noise_val_loader,
        noise_criterion, noise_optimizer, NUM_EPOCHS, device
    )

    print("\nTraining completed!")

if __name__ == "__main__":
    main()
