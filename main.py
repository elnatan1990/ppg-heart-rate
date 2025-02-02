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
    Preprocess PPG (Photoplethysmography) signal using a bandpass filter to isolate heart rate frequencies.

    The function applies a Butterworth bandpass filter to remove frequencies outside the normal heart rate range.
    Normal heart rates are between 40-180 BPM, which corresponds to 0.67-3.0 Hz.

    Parameters:
    -----------
    signal : numpy.ndarray
        Raw PPG signal time series data
    fs : float, default=32.0
        Sampling frequency of the signal in Hz (samples per second)

    Returns:
    --------
    numpy.ndarray
        Filtered PPG signal with frequencies only in the heart rate range

    Technical Details:
    -----------------
    - Uses a 3rd order Butterworth filter
    - Nyquist frequency (fs/2) is used to normalize the cutoff frequencies
    - Low cutoff: 0.67 Hz (40 BPM)
    - High cutoff: 3.0 Hz (180 BPM)
    - filtfilt is used for zero-phase filtering (no time delay)
    """
    # Calculate Nyquist frequency (the highest frequency that can be measured)
    nyquist = fs / 2

    # Convert heart rate frequencies to normalized frequencies for the filter
    L_freq_bpm = 40 / 60
    H_freq_bpm = 180 / 60
    low = L_freq_bpm / nyquist  # 40 BPM = 0.67 Hz
    high = H_freq_bpm / nyquist  # 180 BPM = 3.0 Hz

    # Create Butterworth filter coefficients
    b, a = butter(2, [low, high], btype='band')

    # Apply zero-phase filtering
    return filtfilt(b, a, signal)

def estimate_hr_fft(signal, fs=32.0):
    """
    Estimate heart rate from a PPG signal using Fast Fourier Transform (FFT).

    This function converts the time-domain signal to frequency domain and identifies
    the dominant frequency within the expected heart rate range (40-180 BPM).

    Parameters:
    -----------
    signal : numpy.ndarray
        Preprocessed PPG signal time series data
    fs : float, default=32.0
        Sampling frequency of the signal in Hz (samples per second)

    Returns:
    --------
    float
        Estimated heart rate in beats per minute (BPM)

    Technical Details:
    -----------------
    Process:
    1. Apply FFT to convert signal to frequency domain
    2. Calculate corresponding frequencies for each FFT component
    3. Consider only positive frequencies (negative frequencies are redundant)
    4. Find the strongest frequency component in the heart rate range (0.67-3.0 Hz)
    5. Convert the dominant frequency to BPM by multiplying by 60

    Example:
    --------
    If the dominant frequency is 1.5 Hz:
    Heart Rate = 1.5 Hz * 60 = 90 BPM
    """
    # Apply FFT to get frequency components
    fft_result = fft(signal)

    # Calculate frequency values for each FFT component
    freqs = fftfreq(len(signal), 1 / fs)

    # Keep only positive frequencies (negative are redundant in real signals)
    pos_mask = freqs > 0
    freqs = freqs[pos_mask]
    fft_result = np.abs(fft_result[pos_mask])

    # Find the strongest frequency in heart rate range (0.67-3.0 Hz = 40-180 BPM)
    L_freq_bpm = 40 / 60
    H_freq_bpm = 180 / 60
    hr_mask = (freqs >= L_freq_bpm) & (freqs <= H_freq_bpm)
    dominant_freq = freqs[hr_mask][np.argmax(fft_result[hr_mask])]

    # Convert frequency (Hz) to heart rate (BPM)
    return dominant_freq * 60

def classical_hr_estimation(ppg_signals: pd.DataFrame) -> np.ndarray:
    """
    Estimate heart rate from PPG signals using FFT-based method
    """
    results = []
    for signal in ppg_signals['signal']:
        # Convert string representation to numpy array if needed
        if isinstance(signal, str):
            signal = np.array(eval(signal))

        # Preprocess signal
        cleaned_signal = preprocess_signal(signal)

        # Estimate heart rate
        hr = estimate_hr_fft(cleaned_signal)
        results.append(hr)

    return np.array(results)

def calc_mae(ppg_signals: pd.DataFrame, predicted_hr: np.ndarray) -> tuple[float, pd.DataFrame]:
    """
    Calculate Mean Absolute Error between predicted heart rates and ground truth labels,
    ignoring entries where predicted_hr is -1, and add classical HR predictions to the DataFrame.
    Also reports the number of -1 predictions that were excluded from MAE calculation.

    Parameters:
    -----------
    ppg_signals : pd.DataFrame
        DataFrame containing 'signal' and 'hr' (ground truth) columns
    predicted_hr : np.ndarray
        Array of predicted heart rates from the classical method

    Returns:
    --------
    tuple[float, pd.DataFrame]
        - Mean Absolute Error between predictions and ground truth (excluding -1 predictions)
        - Updated DataFrame with added 'hr_classic' column
    """
    # Add classical HR predictions to the DataFrame
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

def deep_learning_hr_estimation(ppg_signals: pd.DataFrame) -> tuple[np.ndarray, list]:
    """
    Estimate heart rate using deep learning model

    Returns:
    --------
    tuple[np.ndarray, list]
        - Array of predicted heart rates
        - List of validation indices
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    model = PPGNet().to(device)

    # Load checkpoint and extract model state dict
    checkpoint = torch.load('models/hr_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create dataset
    dataset = PPGDataset(ppg_signals['signal'].values)

    # Load the validation split and get indices
    with open('data_splits/hr_split.pkl', 'rb') as f:
        split_data = pickle.load(f)
    val_indices = split_data['val_indices']

    val_loader = load_dataset_split(dataset, 'data_splits/hr_split.pkl', batch_size=32)

    # Make predictions
    predictions = []
    with torch.no_grad():
        for inputs in val_loader:
            inputs = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy().flatten())

    predictions = np.array(predictions)
    return predictions, val_indices

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

def challenge_hr_estimation_noise_model(ppg_signals: pd.DataFrame) -> tuple[np.ndarray, list]:
    """
    Estimate heart rate while identifying noisy signals

    Returns:
    --------
    tuple[np.ndarray, list]
        - Array of predicted heart rates
        - List of validation indices
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

    # Create dataset
    dataset = PPGDataset(ppg_signals['signal'].values)

    # Load the validation split and get indices
    # with open('data_splits/hr_split.pkl', 'rb') as f:
    #     split_data = pickle.load(f)
    # val_indices = split_data['val_indices']
    # val_loader = load_dataset_split(dataset, 'data_splits/hr_split.pkl', batch_size=32)

    val_indices = 0
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

    return np.array(predictions), val_indices

def challenge_hr_estimation_filter_noise(ppg_signals: pd.DataFrame) -> tuple[np.ndarray, list]:
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

    # Create dataset
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

            # Print noise detection stats for this batch
            # noisy_count = np.sum(is_noisy)
            # print(f"Batch {batch_idx}: {noisy_count}/{len(is_noisy)} signals marked as noisy")

            inputs = inputs.to(device)

            # Get HR predictions
            hr_pred = hr_model(inputs)
            hr_pred = hr_pred.cpu().numpy().flatten()

            # Replace predictions for noisy signals with -1
            hr_pred[is_noisy] = -1

            predictions.extend(hr_pred)

    return np.array(predictions), 0

def challenge_hr_estimation_filter_noise_enhanced(ppg_signals: pd.DataFrame) -> tuple[np.ndarray, list]:
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

    # Create dataset
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

    return np.array(predictions), 0

def ensemble_hr_estimation(ppg_signals: pd.DataFrame) -> tuple[np.ndarray, list]:
    """
    Combine deep learning and classical methods for heart rate estimation using only validation indices.

    Parameters:
    -----------
    ppg_signals : pd.DataFrame
        DataFrame containing PPG signals and ground truth heart rates

    Returns:
    --------
    tuple[np.ndarray, list]
        - Array of predicted heart rates (ensemble predictions)
        - List of validation indices
    """
    # Get predictions from deep learning model (this already returns validation predictions)
    deep_learning_predictions, val_indices = deep_learning_hr_estimation(ppg_signals)

    # Get predictions from classical method only for validation indices
    val_signals = ppg_signals.iloc[val_indices]
    classical_predictions = classical_hr_estimation(val_signals)

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

    return ensemble_predictions, val_indices

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
        print('MAE: ', mae) # MAE: 5.0112714771910865

    # Call deep learning method - Train model over HR split into train and validation sets, prediction on validation set only:
    if selector == 2:
        input_file = "ppg_samples_with_hr.parquet"
        ppg_signals = pd.read_parquet(input_file)
        deep_learning_hr, val_indices = deep_learning_hr_estimation(ppg_signals)
        mae, val_ppg_signals = calc_mae_for_val(ppg_signals, deep_learning_hr, val_indices)
        print(val_ppg_signals.head())
        print('MAE: ', mae) # MAE:  4.968111440155171

    # Call ensemble method - Ensemble of classical method with deep learning method (mean prediction improved):
    if selector == 3:
        input_file = "ppg_samples_with_hr.parquet"
        ppg_signals = pd.read_parquet(input_file)
        ensemble_hr, val_indices = ensemble_hr_estimation(ppg_signals)
        mae, val_ppg_signals = calc_mae_for_val(ppg_signals, ensemble_hr, val_indices)
        print(val_ppg_signals.head())
        print('MAE: ', mae) # 4.229609519144526

    ### Challenge 2 ###
    # Call challenge method - Noise model created by labeling signals into noisy and regular signals by 80:20 ratio:
    if selector == 4:
        challenge_file = "ppg_samples_with_hr_v4.parquet"
        ppg_signals = pd.read_parquet(challenge_file)
        challenge_hr, val_indices = challenge_hr_estimation_noise_model(ppg_signals)
        mae, ppg_signals = calc_mae(ppg_signals, challenge_hr)
        print(ppg_signals.head())
        print('MAE: ', mae) # MAE: 11.535496235178407

    # Call challenge method - Noise estimation by noise_level = np.std(np.diff(signal)) and filtering by 80:20 ratio:
    if selector == 5:
        challenge_file = "ppg_samples_with_hr_v4.parquet"
        ppg_signals = pd.read_parquet(challenge_file)
        challenge_hr, val_indices = challenge_hr_estimation_filter_noise(ppg_signals)
        mae, ppg_signals = calc_mae(ppg_signals, challenge_hr)
        print(ppg_signals.head())
        print('MAE: ', mae) # MAE: 7.087473433401042

    # Call challenge method - Ensemble of classical method with deep learning method (mean prediction improved):
    if selector == 6:
        challenge_file = "ppg_samples_with_hr_v4.parquet"
        ppg_signals = pd.read_parquet(challenge_file)
        challenge_hr, val_indices = challenge_hr_estimation_filter_noise_enhanced(ppg_signals)
        mae, ppg_signals = calc_mae(ppg_signals, challenge_hr)
        print(ppg_signals.head())
        print('MAE: ', mae) # MAE: 6.778399694236167
