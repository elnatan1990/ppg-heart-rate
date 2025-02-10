import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
from torch.utils.data import DataLoader
from models import PPGNet, NoiseClassifier, PPGDataset
from dataset_utils import load_dataset_split
import pickle

def preprocess_signal(signal, fs=32.0):
    """
    Preprocess PPG signal using optimized parameters from grid search.
    """
    # Calculate Nyquist frequency
    nyquist = fs / 2

    # Use optimized parameters from grid search
    L_freq_bpm = 37.0 / 60  # Optimized low frequency
    H_freq_bpm = 178.0 / 60  # Optimized high frequency
    low = L_freq_bpm / nyquist
    high = H_freq_bpm / nyquist

    # Use optimized filter order = 1
    b, a = butter(1, [low, high], btype='band')

    # Apply zero-phase filtering
    return filtfilt(b, a, signal)

def estimate_hr_fft(signal, fs=32.0):
    """
    Estimate heart rate using optimized FFT parameters from grid search.
    Best parameters found:
    - padding_factor: 2.5
    - low_freq_bpm: 37.0
    - high_freq_bpm: 178.0
    Expected MAE: ~3.24
    """
    # Apply FFT with optimized padding factor
    padding_factor = 2.5  # Optimized padding
    n_pad = int(len(signal) * padding_factor)
    fft_result = fft(signal, n=n_pad)
    freqs = fftfreq(n_pad, 1 / fs)

    # Keep only positive frequencies
    pos_mask = freqs > 0
    freqs = freqs[pos_mask]
    fft_result = np.abs(fft_result[pos_mask])

    # Use optimized frequency range
    L_freq_bpm = 37.0 / 60  # Optimized value
    H_freq_bpm = 178.0 / 60  # Optimized value
    hr_mask = (freqs >= L_freq_bpm) & (freqs <= H_freq_bpm)

    if not any(hr_mask):  # Safety check if no frequencies in range
        return 75.0  # Return typical heart rate

    dominant_freq = freqs[hr_mask][np.argmax(fft_result[hr_mask])]
    hr = dominant_freq * 60

    # Additional safety check for physiological range
    if hr < 37.0 or hr > 178.0:
        return 75.0

    return hr

def classical_hr_estimation(ppg_signals: pd.DataFrame) -> np.ndarray:
    """
    Estimate heart rate using optimized parameters from grid search
    """
    results = []
    for signal in ppg_signals['signal']:
        if isinstance(signal, str):
            signal = np.array(eval(signal))

        # Use optimized preprocessing and estimation
        cleaned_signal = preprocess_signal(signal)
        hr = estimate_hr_fft(cleaned_signal)
        results.append(hr)

    return np.array(results)

def calc_mae(ppg_signals: pd.DataFrame, predicted_hr: np.ndarray) -> tuple[float, pd.DataFrame]:
    """
    Calculate Mean Absolute Error between predicted heart rates and ground truth labels,
    ignoring entries where predicted_hr is -1

    Parameters:
    -----------
    ppg_signals : pd.DataFrame
        DataFrame containing 'signal' and 'hr' (ground truth) columns
    predicted_hr : np.ndarray
        Array of predicted heart rates

    Returns:
    --------
    tuple[float, pd.DataFrame]
        - Mean Absolute Error between predictions and ground truth
        - Updated DataFrame with added predictions
    """
    # Add predictions to the DataFrame
    ppg_signals = ppg_signals.copy()
    ppg_signals['hr_pred'] = predicted_hr

    # Count total predictions and number of -1s
    total_predictions = len(predicted_hr)
    invalid_predictions = np.sum(predicted_hr == -1)
    valid_predictions = total_predictions - invalid_predictions

    # Print statistics
    print(f"Total predictions: {total_predictions}")
    print(f"Invalid predictions (-1): {invalid_predictions} ({(invalid_predictions/total_predictions)*100:.2f}%)")
    print(f"Valid predictions: {valid_predictions} ({(valid_predictions/total_predictions)*100:.2f}%)")

    # Create mask for valid predictions (not -1)
    valid_mask = ppg_signals['hr_pred'] != -1

    # Calculate absolute error only for valid predictions
    absolute_errors = np.abs(ppg_signals.loc[valid_mask, 'hr'] - ppg_signals.loc[valid_mask, 'hr_pred'])

    # Calculate mean absolute error
    mae = np.mean(absolute_errors)

    return mae, ppg_signals

def calc_mae_for_val(ppg_signals: pd.DataFrame, predicted_hr: np.ndarray, val_indices: list) -> tuple[float, pd.DataFrame]:
    """
    Calculate Mean Absolute Error between predicted heart rates and ground truth labels,
    only for the validation set indices.

    Parameters:
    -----------
    ppg_signals : pd.DataFrame
        DataFrame containing 'signal' and 'hr' (ground truth) columns
    predicted_hr : np.ndarray
        Array of predicted heart rates from the model
    val_indices : list
        List of validation set indices

    Returns:
    --------
    tuple[float, pd.DataFrame]
        - Mean Absolute Error between predictions and ground truth
        - DataFrame containing only validation set rows with predictions
    """
    # Create a copy of only the validation rows
    val_ppg_signals = ppg_signals.iloc[val_indices].copy()

    # Add predictions
    val_ppg_signals['hr_pred'] = predicted_hr

    # Count total predictions and number of -1s
    total_predictions = len(predicted_hr)
    invalid_predictions = np.sum(predicted_hr == -1)
    valid_predictions = total_predictions - invalid_predictions

    # Print statistics
    print(f"Total predictions: {total_predictions}")
    print(f"Invalid predictions (-1): {invalid_predictions} ({(invalid_predictions / total_predictions) * 100:.2f}%)")
    print(f"Valid predictions: {valid_predictions} ({(valid_predictions / total_predictions) * 100:.2f}%)")

    # Create mask for valid predictions (not -1)
    valid_mask = val_ppg_signals['hr_pred'] != -1

    # Calculate absolute error only for valid predictions
    absolute_errors = np.abs(val_ppg_signals.loc[valid_mask, 'hr'] - val_ppg_signals.loc[valid_mask, 'hr_pred'])

    # Calculate mean absolute error
    mae = np.mean(absolute_errors)

    return mae, val_ppg_signals

def deep_learning_hr_estimation(ppg_signals: pd.DataFrame) -> np.ndarray:
    """
    Estimate heart rate using deep learning model for the full dataset

    Returns:
    --------
    np.ndarray
        Array of predicted heart rates
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    model = PPGNet().to(device)

    # Load checkpoint and extract model state dict
    checkpoint = torch.load('models/hr_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create dataset and dataloader for full dataset
    dataset = PPGDataset(ppg_signals['signal'].values)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    # Make predictions
    predictions = []
    with torch.no_grad():
        for inputs in dataloader:
            inputs = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy().flatten())

    return np.array(predictions)

def ensemble_hr_estimation(ppg_signals: pd.DataFrame) -> np.ndarray:
    """
    Combine deep learning and classical methods for heart rate estimation for the full dataset.

    Parameters:
    -----------
    ppg_signals : pd.DataFrame
        DataFrame containing PPG signals and ground truth heart rates

    Returns:
    --------
    np.ndarray
        Array of predicted heart rates (ensemble predictions)
    """
    # Get predictions from both methods for the full dataset
    deep_learning_predictions = deep_learning_hr_estimation(ppg_signals)
    classical_predictions = classical_hr_estimation(ppg_signals)

    # Now both prediction arrays should be the same size
    assert len(deep_learning_predictions) == len(classical_predictions), "Prediction arrays must be same size"

    # Initialize ensemble predictions array
    ensemble_predictions = np.zeros_like(deep_learning_predictions)

    # For each prediction
    for i in range(len(deep_learning_predictions)):
        # If deep learning prediction is valid (not -1)
        if deep_learning_predictions[i] != -1:
            # Take average of both predictions
            ensemble_predictions[i] = (deep_learning_predictions[i] + classical_predictions[i]) / 2
        else:
            # If deep learning marks it as noisy, use classical prediction
            ensemble_predictions[i] = classical_predictions[i]

    return ensemble_predictions

def detect_noise(signal):
    """
    Detect if a PPG signal is noisy using statistical features.
    Returns True if signal is noisy, False otherwise.
    """
    if isinstance(signal, str):
        try:
            signal = eval(signal)
        except:
            signal = eval(signal.replace('\n', ''))
    signal = np.array(signal)

    # Calculate noise level using signal derivative variance
    noise_level = np.std(np.diff(signal))
    return noise_level

def get_noise_threshold(signals):
    """
    Calculate the noise threshold based on the 80th percentile of all signals
    """
    noise_levels = [detect_noise(signal) for signal in signals]
    threshold = np.percentile(noise_levels, 80)  # 80th percentile for 20% noisy signals
    return threshold

def challenge_hr_estimation_noise_model(ppg_signals: pd.DataFrame) -> np.ndarray:
    """
    Estimate heart rate while identifying noisy signals using deep learning noise model

    Returns:
    --------
    np.ndarray: Array of predicted heart rates (-1 for noisy signals)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize models
    hr_model = PPGNet().to(device)
    noise_model = NoiseClassifier().to(device)

    # Load checkpoints and extract model state dicts
    hr_checkpoint = torch.load('models/hr_model.pth', map_location=device)
    noise_checkpoint = torch.load('models/noise_model.pth', map_location=device)

    hr_model.load_state_dict(hr_checkpoint['model_state_dict'])
    noise_model.load_state_dict(noise_checkpoint['model_state_dict'])

    hr_model.eval()
    noise_model.eval()

    # Create dataset and dataloader for full dataset
    dataset = PPGDataset(ppg_signals['signal'].values)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    # Make predictions
    predictions = []
    with torch.no_grad():
        for inputs in dataloader:
            inputs = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
            inputs = inputs.to(device)

            # Check if signal is too noisy
            noise_pred = noise_model(inputs)
            is_noisy = torch.argmax(noise_pred, dim=1)

            # Get HR predictions
            hr_pred = hr_model(inputs)

            # Replace predictions for noisy signals with -1
            hr_pred[is_noisy == 1] = -1

            # Mark predictions outside physiological range as noisy (-1)
            noisy_mask = (hr_pred < 60) | (hr_pred > 150)
            hr_pred[noisy_mask] = -1

            predictions.extend(hr_pred.cpu().numpy().flatten())

    return np.array(predictions)

def challenge_hr_estimation_filter_noise(ppg_signals: pd.DataFrame) -> np.ndarray:
    """
    Estimate heart rate while identifying noisy signals using statistical approach
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize HR model
    hr_model = PPGNet().to(device)

    # Load checkpoint and extract model state dict
    hr_checkpoint = torch.load('models/hr_model.pth', map_location=device)
    hr_model.load_state_dict(hr_checkpoint['model_state_dict'])
    hr_model.eval()

    # Create dataset and dataloader for full dataset
    dataset = PPGDataset(ppg_signals['signal'].values)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    # Calculate noise threshold from original signals (not transformed)
    noise_threshold = get_noise_threshold(ppg_signals['signal'].values)
    print(f"Calculated noise threshold: {noise_threshold}")

    # Make predictions
    predictions = []
    with torch.no_grad():
        for batch_idx, inputs in enumerate(dataloader):
            inputs = inputs[0] if isinstance(inputs, (list, tuple)) else inputs

            # Get the original signals for this batch from the DataFrame
            batch_start = batch_idx * dataloader.batch_size
            batch_end = min((batch_idx + 1) * dataloader.batch_size, len(ppg_signals))
            batch_signals = ppg_signals['signal'].iloc[batch_start:batch_end].values

            # Check noise levels using original signals
            is_noisy = np.array([
                detect_noise(signal) >= noise_threshold
                for signal in batch_signals
            ])

            inputs = inputs.to(device)

            # Get HR predictions
            hr_pred = hr_model(inputs)
            hr_pred = hr_pred.cpu().numpy().flatten()

            # Replace predictions for noisy signals with -1
            hr_pred[is_noisy] = -1

            predictions.extend(hr_pred)

    return np.array(predictions)

def challenge_hr_estimation_ensemble(ppg_signals: pd.DataFrame) -> np.ndarray:
    """
    Estimate heart rate while identifying noisy signals using statistical approach
    and combining with classical method predictions for better accuracy.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize HR model
    hr_model = PPGNet().to(device)

    # Load checkpoint and extract model state dict
    hr_checkpoint = torch.load('models/hr_model.pth', map_location=device)
    hr_model.load_state_dict(hr_checkpoint['model_state_dict'])
    hr_model.eval()

    # Create dataset and dataloader for full dataset
    dataset = PPGDataset(ppg_signals['signal'].values)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    # Calculate noise threshold from original signals (not transformed)
    noise_threshold = get_noise_threshold(ppg_signals['signal'].values)
    print(f"Calculated noise threshold: {noise_threshold}")

    # Get classical method predictions for all signals
    classical_predictions = classical_hr_estimation(ppg_signals)

    # Make predictions
    predictions = []
    with torch.no_grad():
        for batch_idx, inputs in enumerate(dataloader):
            inputs = inputs[0] if isinstance(inputs, (list, tuple)) else inputs

            # Get the original signals for this batch from the DataFrame
            batch_start = batch_idx * dataloader.batch_size
            batch_end = min((batch_idx + 1) * dataloader.batch_size, len(ppg_signals))
            batch_signals = ppg_signals['signal'].iloc[batch_start:batch_end].values

            # Get the corresponding classical predictions for this batch
            batch_classical = classical_predictions[batch_start:batch_end]

            # Check noise levels using original signals
            is_noisy = np.array([
                detect_noise(signal) >= noise_threshold
                for signal in batch_signals
            ])

            inputs = inputs.to(device)

            # Get HR predictions from deep learning model
            hr_pred = hr_model(inputs)
            hr_pred = hr_pred.cpu().numpy().flatten()

            # For filtered signals (not noisy), use mean of both methods
            final_predictions = np.copy(hr_pred)
            clean_mask = ~is_noisy
            final_predictions[clean_mask] = (hr_pred[clean_mask] + batch_classical[clean_mask]) / 2
            final_predictions[is_noisy] = -1

            predictions.extend(final_predictions)

    return np.array(predictions)

def ensemble_hr_estimation(ppg_signals: pd.DataFrame) -> np.ndarray:
    """
    Combine deep learning and classical methods for heart rate estimation for the full dataset.

    Parameters:
    -----------
    ppg_signals : pd.DataFrame
        DataFrame containing PPG signals and ground truth heart rates

    Returns:
    --------
    np.ndarray
        Array of predicted heart rates (ensemble predictions)
    """
    # Get predictions from both methods for the full dataset
    deep_learning_predictions = deep_learning_hr_estimation(ppg_signals)
    classical_predictions = classical_hr_estimation(ppg_signals)

    # Now both prediction arrays should be the same size
    assert len(deep_learning_predictions) == len(classical_predictions), "Prediction arrays must be same size"

    # Initialize ensemble predictions array
    ensemble_predictions = np.zeros_like(deep_learning_predictions)

    # For each prediction
    for i in range(len(deep_learning_predictions)):
        # If deep learning prediction is valid (not -1)
        if deep_learning_predictions[i] != -1:
            # Take average of both predictions
            ensemble_predictions[i] = (deep_learning_predictions[i] + classical_predictions[i]) / 2
        else:
            # If deep learning marks it as noisy, use classical prediction
            ensemble_predictions[i] = classical_predictions[i]

    return ensemble_predictions

if __name__ == "__main__":
    selector = 1

    ### Challenge 1 ###
    # Call classical method - Estimate heart rate from PPG signals using FFT-based method:
    if selector == 1:
        input_file = "ppg_samples_with_hr.parquet"
        ppg_signals = pd.read_parquet(input_file)
        classical_hr = classical_hr_estimation(ppg_signals)
        mae, ppg_signals = calc_mae(ppg_signals, classical_hr)
        print(ppg_signals.head())
        print('MAE: ', mae) # MAE: 5.0112714771910865 improved to 3.236645271446222

    # Call deep learning method - Train model over HR split into train and validation sets, prediction on validation set only:
    if selector == 2:
        input_file = "ppg_samples_with_hr.parquet"
        ppg_signals = pd.read_parquet(input_file)
        deep_learning_hr = deep_learning_hr_estimation(ppg_signals)
        mae, ppg_signals = calc_mae(ppg_signals, deep_learning_hr)
        print(ppg_signals.head())
        print('MAE: ', mae) # MAE:  4.620325068482422

    # Call ensemble method - Ensemble of classical method with deep learning method (mean prediction improved):
    if selector == 3:
        input_file = "ppg_samples_with_hr.parquet"
        ppg_signals = pd.read_parquet(input_file)
        ensemble_hr = ensemble_hr_estimation(ppg_signals)
        mae, ppg_signals = calc_mae(ppg_signals, ensemble_hr)
        print(ppg_signals.head())
        print('MAE: ', mae) # MAE: 3.827082710975075 improved to 3.3374703393987555

    ### Challenge 2 ###
    # Call challenge method - Noise model created by labeling signals into noisy and regular signals by 80:20 ratio:
    if selector == 4:
        challenge_file = "ppg_samples_with_hr_v4.parquet"
        ppg_signals = pd.read_parquet(challenge_file)
        challenge_hr = challenge_hr_estimation_noise_model(ppg_signals)
        mae, ppg_signals = calc_mae(ppg_signals, challenge_hr)
        print(ppg_signals.head())
        print('MAE: ', mae) # MAE: 11.535496235178407

    # Call challenge method - Noise estimation by noise_level = np.std(np.diff(signal)) and filtering by 80:20 ratio:
    if selector == 5:
        challenge_file = "ppg_samples_with_hr_v4.parquet"
        ppg_signals = pd.read_parquet(challenge_file)
        challenge_hr = challenge_hr_estimation_filter_noise(ppg_signals)
        mae, ppg_signals = calc_mae(ppg_signals, challenge_hr)
        print(ppg_signals.head())
        print('MAE: ', mae) # MAE: 7.087473433401042

    # Call challenge method - Ensemble of classical method with deep learning method (mean prediction improved):
    if selector == 6:
        challenge_file = "ppg_samples_with_hr_v4.parquet"
        ppg_signals = pd.read_parquet(challenge_file)
        challenge_hr = challenge_hr_estimation_ensemble(ppg_signals)
        mae, ppg_signals = calc_mae(ppg_signals, challenge_hr)
        print(ppg_signals.head())
        print('MAE: ', mae) # MAE: 6.778399694236167 improved to 6.324337772605617 with classical method
