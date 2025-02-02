import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt


class PPGDataset(Dataset):
    """Dataset class for PPG signals"""

    def __init__(self, signals, labels=None, transform=True):
        self.signals = signals
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        # Get signal
        signal = self.signals[idx]

        # Convert string to numpy array if needed
        if isinstance(signal, str):
            try:
                signal = eval(signal)
            except:
                signal = eval(signal.replace('\n', ''))
            signal = np.array(signal, dtype=np.float32)

        # Make sure it's a numpy array
        if not isinstance(signal, np.ndarray):
            signal = np.array(signal, dtype=np.float32)

        # Ensure the array is contiguous and has the right format
        signal = np.array(signal, dtype=np.float32, order='C').copy()

        # Apply preprocessing if enabled
        if self.transform:
            signal = self.preprocess_signal(signal)

        # Convert to tensor (should now work with the copied, contiguous array)
        signal = torch.from_numpy(signal).float()

        # Add channel dimension
        signal = signal.unsqueeze(0)

        if self.labels is not None:
            label = float(self.labels[idx])
            return signal, torch.tensor([label], dtype=torch.float32)
        return signal

    def preprocess_signal(self, signal, fs=32.0):
        """Preprocess PPG signal using bandpass filter"""
        nyquist = fs / 2
        low = 0.67 / nyquist
        high = 3.0 / nyquist
        b, a = butter(3, [low, high], btype='band')
        filtered = filtfilt(b, a, signal)
        return filtered.astype(np.float32)


class PPGNet(nn.Module):
    """1D CNN for heart rate estimation from PPG signals"""

    def __init__(self):
        super(PPGNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 64, kernel_size=15, padding=7)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=9, padding=4)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, padding=2)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)

        # Dropout
        self.dropout = nn.Dropout(0.5)

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # Convolutional blocks
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2)

        x = F.relu(self.bn3(self.conv3(x)))

        # Global average pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x


class NoiseClassifier(nn.Module):
    """CNN for classifying noisy PPG signals"""

    def __init__(self):
        super(NoiseClassifier, self).__init__()

        # Convolutional layers with residual connections
        self.conv1 = nn.Conv1d(1, 64, kernel_size=15, padding=7)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=15, padding=7)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=9, padding=4)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=9, padding=4)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(128)

        # Dropout
        self.dropout = nn.Dropout(0.5)

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 2)  # Binary classification: noisy vs clean

    def forward(self, x):
        # First residual block
        identity = self.conv1(x)
        x = F.relu(self.bn1(identity))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x + identity
        x = F.max_pool1d(x, 2)

        # Second residual block
        identity = self.conv3(x)
        x = F.relu(self.bn3(identity))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x + identity
        x = F.max_pool1d(x, 2)

        # Global average pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x


class NoiseDataset(Dataset):
    """Dataset class for noise classification with 80-20 split"""

    def __init__(self, signals):
        self.signals = signals['signal'].values
        # Calculate noise labels based on 80-20 distribution
        self.labels = self._calculate_noise_labels(signals)

    def __len__(self):
        return len(self.signals)

    def _calculate_noise_labels(self, signals):
        """Calculate binary noise labels using 80th percentile threshold"""
        noise_levels = []
        for signal in signals['signal']:
            if isinstance(signal, str):
                try:
                    signal = eval(signal)
                except:
                    signal = eval(signal.replace('\n', ''))
            signal = np.array(signal)

            # Calculate noise level using signal variance and derivative
            noise_level = np.std(np.diff(signal))
            noise_levels.append(noise_level)

        # Convert to numpy array
        noise_levels = np.array(noise_levels)

        # Find the 80th percentile threshold
        threshold = np.percentile(noise_levels, 80)

        # Create binary labels (0 for clean, 1 for noisy)
        # This will automatically give us 80% clean (0) and 20% noisy (1)
        labels = (noise_levels >= threshold).astype(int)

        return labels

    def __getitem__(self, idx):
        # Get signal
        signal = self.signals[idx]

        # Convert string to numpy array if needed
        if isinstance(signal, str):
            try:
                signal = eval(signal)
            except:
                signal = eval(signal.replace('\n', ''))
            signal = np.array(signal, dtype=np.float32)

        # Ensure the array is contiguous and has the right format
        signal = np.array(signal, dtype=np.float32, order='C').copy()

        # Convert to tensor
        signal = torch.from_numpy(signal).float()

        # Add channel dimension
        signal = signal.unsqueeze(0)

        # Get label (ensure it's a single integer)
        label = int(self.labels[idx])
        return signal, label


class ImprovedPPGNet(nn.Module):
    """Improved 1D CNN for heart rate estimation"""

    def __init__(self, dropout_rate=0.3):
        super(ImprovedPPGNet, self).__init__()

        # Initial convolution block
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Residual blocks
        self.res1 = ResidualBlock(64, 128)
        self.res2 = ResidualBlock(128, 256)
        self.res3 = ResidualBlock(256, 512)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv1d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Global pooling and fully connected layers
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)

        # Residual blocks
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)

        # Apply attention
        attention_weights = self.attention(x)
        x = x * attention_weights

        # Global pooling and fully connected layers
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResidualBlock(nn.Module):
    """Residual block with bottleneck design"""

    def __init__(self, in_channels, out_channels, dropout_rate=0.3):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels // 2, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(out_channels // 2)

        self.conv2 = nn.Conv1d(out_channels // 2, out_channels // 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels // 2)

        self.conv3 = nn.Conv1d(out_channels // 2, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(dropout_rate)

        # Shortcut connection
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        identity = self.shortcut(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.bn3(self.conv3(x))

        x += identity
        x = F.relu(x)

        return x
