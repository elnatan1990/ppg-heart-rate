import pandas as pd
import numpy as np
from itertools import product
from tqdm import tqdm
import json
from datetime import datetime
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq


def batch_process_signals(signals, params, fs=32.0):
    """Process multiple signals at once for better performance"""
    results = []
    nyquist = fs / 2
    low = params['low_freq_bpm'] / 60 / nyquist
    high = params['high_freq_bpm'] / 60 / nyquist

    # Create filter once
    b, a = butter(params['filter_order'], [low, high], btype='band')

    for signal in signals:
        if isinstance(signal, str):
            signal = np.array(eval(signal))

        # Apply bandpass filter
        filtered = filtfilt(b, a, signal)

        # Compute FFT with optimal padding
        n_pad = int(len(filtered) * params['padding_factor'])
        fft_result = fft(filtered, n=n_pad)
        freqs = fftfreq(n_pad, 1 / fs)

        # Process positive frequencies
        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        fft_result = np.abs(fft_result[pos_mask])

        # Find dominant frequency in HR range
        hr_mask = (freqs >= params['low_freq_bpm'] / 60) & (freqs <= params['high_freq_bpm'] / 60)
        masked_freqs = freqs[hr_mask]
        masked_spectrum = fft_result[hr_mask]

        if len(masked_spectrum) > 0:
            dominant_freq = masked_freqs[np.argmax(masked_spectrum)]
            results.append(dominant_freq * 60)
        else:
            results.append(75)  # fallback

    return np.array(results)


def optimized_grid_search(ppg_signals, param_grid):
    """Optimized grid search focusing on key parameters"""
    results = []

    # Pre-load all signals
    signals = ppg_signals['signal'].values
    true_hr = ppg_signals['hr'].values

    # Generate parameter combinations
    param_combinations = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]

    for params in tqdm(param_combinations, desc="Optimized Grid Search"):
        try:
            # Process all signals in batch
            predictions = batch_process_signals(signals, params)

            # Calculate MAE
            mae = np.mean(np.abs(predictions - true_hr))

            results.append({
                'params': params,
                'mae': mae
            })
            print(f"Parameters: {params}, MAE: {mae}")

        except Exception as e:
            print(f"Error with parameters {params}: {str(e)}")
            continue

    if results:
        results.sort(key=lambda x: x['mae'])
        return results
    return None


if __name__ == "__main__":
    # Load data
    input_file = "ppg_samples_with_hr.parquet"
    ppg_signals = pd.read_parquet(input_file)

    # Focused parameter grid based on previous results
    optimized_param_grid = {
        'filter_order': [1],  # 1 performed best
        'low_freq_bpm': [36.75, 37.0, 37.25],  # Focus around 37.0
        'high_freq_bpm': [177.0, 177.5, 178.0],  # Focus around 177.5
        'padding_factor': [2.25, 2.5, 2.75]  # Focus around 2.5
    }

    print("\nRunning optimized grid search...")
    results = optimized_grid_search(ppg_signals, optimized_param_grid)

    if results:
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f'optimized_grid_search_results_{timestamp}.json', 'w') as f:
            json.dump(results, f, indent=4)

        print("\nBest parameters:")
        print(f"Parameters: {results[0]['params']}")
        print(f"MAE: {results[0]['mae']}")

        print("\nPrevious best MAE: 3.2519807911091614")
        print(f"Improvement: {3.2519807911091614 - results[0]['mae']}")

        # Best parameters:
        # {'filter_order': 1, 'low_freq_bpm': 37.0, 'high_freq_bpm': 178.0, 'padding_factor': 2.5}
        # MAE: 3.236645271446222
    else:
        print("\nNo successful results found")
