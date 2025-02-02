# PPG Heart Rate Estimation

This project implements various methods for heart rate estimation from Photoplethysmography (PPG) signals, combining classical signal processing with deep learning approaches. The system includes noise detection and handling capabilities for robust heart rate estimation in real-world conditions.

## Project Overview

The project presents three main approaches for heart rate estimation:
1. Classical FFT-based method
2. Deep learning using PPGNet
3. Ensemble method combining both approaches

Additionally, it includes two noise handling strategies:
- Deep learning-based noise classification
- Statistical noise detection

## Setup Instructions

### Dependencies

Required Python packages:
```python
torch>=1.9.0
torchvision>=0.10.0
torchaudio>=0.9.0
scipy>=1.7.0
pandas>=1.4.0
numpy>=1.20.0
pyarrow>=7.0.0
fastparquet>=0.8.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
```

### Quick Installation

Run the following commands to set up your environment:

```bash
# Install PyTorch and related packages
pip3 install torch torchvision torchaudio

# Install data processing packages
pip install scipy pyarrow fastparquet

# Install other required packages
pip install pandas numpy scikit-learn matplotlib
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ppg-heart-rate
cd ppg-heart-rate
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download pre-trained models:
```bash
mkdir models
# Download hr_model.pth and noise_model.pth into models/
```

## Technical Approach

### Classical Method (MAE: 5.01)

The classical approach uses signal processing techniques:
- Butterworth bandpass filter (0.67-3.0 Hz) for preprocessing
- FFT for frequency domain analysis
- Peak detection in physiological heart rate range (40-180 BPM)

```python
def preprocess_signal(signal, fs=32.0):
    nyquist = fs / 2
    low = (40/60) / nyquist   # 40 BPM
    high = (180/60) / nyquist # 180 BPM
    b, a = butter(2, [low, high], btype='band')
    return filtfilt(b, a, signal)
```

### Deep Learning Method (MAE: 4.97)

Uses a custom PPGNet architecture:
- Convolutional layers for feature extraction
- Batch normalization for training stability
- Dropout for regularization
- Dense layers for final prediction

Training parameters:
- Learning rate: 0.001
- Batch size: 32
- Optimizer: Adam
- Loss function: Mean Absolute Error

### Ensemble Method (MAE: 4.23)

Combines predictions from both approaches:
- Averages predictions when deep learning output is valid
- Falls back to classical method for noisy signals
- Achieves better accuracy than either method alone

### Noise Handling

Two strategies implemented:

1. Model-based (NoiseClassifier):
- Binary classification (clean/noisy)
- 80:20 ratio assumption for data distribution
- MAE: 11.54 (high due to conservative noise detection)

2. Statistical approach:
- Uses signal derivative variance
- Adaptive threshold at 80th percentile
- MAE: 7.09
- Enhanced version with ensemble: MAE: 6.78

## Ensemble Method Analysis

### Why Ensemble Methods Work

The project's ensemble approach combines classical signal processing with deep learning, achieving better results (MAE: 4.23) than either method alone (Classical: 5.01, Deep Learning: 4.97). This improvement demonstrates that "the whole is greater than the sum of its parts" for several reasons:

1. **Complementary Strengths**
   - Classical Method: Strong at frequency analysis and robust to certain types of noise
   - Deep Learning: Excels at pattern recognition and handling non-linear relationships
   - Combined: Each method compensates for the other's weaknesses

2. **Error Diversity**
   - Different approaches make different types of mistakes
   - Classical method might struggle with complex patterns
   - Deep learning might be sensitive to unusual signal shapes
   - Averaging predictions reduces the impact of individual errors

3. **Feature Coverage**
   - Classical Method: Focuses on frequency-domain features
   - Deep Learning: Learns time-domain and abstract features
   - Together: More comprehensive signal analysis

4. **Robustness to Noise**
   - Classical Method: Strong theoretical foundation for regular patterns
   - Deep Learning: Better at handling irregular patterns and artifacts
   - Combined: More reliable across different signal qualities

### Performance Metrics

| Challenge | Method | MAE (beats/min) | Improvement |
|-----------|--------|----------------|-------------|
| Challenge 1 | Classical | 5.01 | Baseline |
| Challenge 1 | Deep Learning | 4.97 | -0.8% |
| Challenge 1 | Ensemble | 4.23 | -15.6% |
| Challenge 2 | Noise Model | 11.54 | Baseline |
| Challenge 2 | Statistical Noise | 7.09 | -38.6% |
| Challenge 2 | Enhanced Ensemble | 6.78 | -41.2% |

The ensemble method shows significant improvement over individual approaches:
- 15.6% reduction in MAE compared to classical method
- 14.9% reduction in MAE compared to deep learning
- More stable predictions across different signal qualities

For noise handling:
- Statistical approach improves MAE by 38.6% over baseline
- Enhanced ensemble method further improves to 41.2% reduction
- Shows effectiveness of statistical noise detection over model-based approach

## Future Improvements

1. Model Architecture:
   - Experiment with different architectures (LSTM, Transformer)
   - Implement attention mechanisms
   - Try transfer learning from larger datasets

2. Noise Handling:
   - Develop more sophisticated noise detection methods
   - Implement signal quality index (SQI)
   - Explore adaptive filtering techniques

3. Data Processing:
   - Implement real-time processing capabilities
   - Add motion artifact correction
   - Explore wavelet-based analysis

4. Validation:
   - Expand testing on diverse datasets
   - Add cross-validation
   - Implement confidence metrics

## Training

### Overview

The training pipeline includes:
- Heart rate estimation model (PPGNet)
- Noise classification model (NoiseClassifier)
- Custom datasets and data loaders
- Early stopping and model checkpointing

### Training Parameters

```python
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_EPOCHS = 50
```

### Model Architecture

1. PPGNet (Heart Rate Estimation):
   - Convolutional layers for feature extraction
   - Batch normalization for stable training
   - Dropout for regularization
   - Trained with MSE loss

2. NoiseClassifier:
   - Binary classification (clean/noisy signals)
   - Similar architecture to PPGNet
   - Trained with CrossEntropy loss

### Training Process

1. Data Loading and Splitting:
```python
train_loader, val_loader = create_dataloaders(
    ppg_signals, 
    batch_size=32, 
    val_split=0.1
)
```

2. Training Loop Features:
   - Early stopping with patience=5
   - Best model checkpointing
   - Validation after each epoch
   - Comprehensive metrics logging

3. Model Evaluation:
   - MAE for heart rate estimation
   - Accuracy for noise classification
   - Validation loss tracking

### Training Script Usage

Run the training script with default parameters:
```bash
python train.py
```

Custom parameters:
```bash
python train.py --batch_size 64 --learning_rate 0.001 --epochs 100
```

### Model Checkpoints

Models are saved automatically:
- `models/hr_model.pth`: Best heart rate estimation model
- `models/noise_model.pth`: Best noise classification model

### Training Metrics

Training progress is logged with:
- Training loss
- Validation loss
- MAE (heart rate) or Accuracy (noise)
- Early stopping status

## Usage

Run different estimation methods using the selector variable:

```python
# Classical method
python main.py --selector 1

# Deep learning method
python main.py --selector 2

# Ensemble method
python main.py --selector 3

# Noise handling methods
python main.py --selector 4  # Model-based
python main.py --selector 5  # Statistical
python main.py --selector 6  # Enhanced statistical
```

## Model Architectures

### PPGNet (Heart Rate Estimation)
```
Architecture:
- Input: 1D PPG signal
- Conv1D layers: (1→64→128→256 channels)
- Batch Normalization after each conv layer
- Max Pooling after first two conv layers
- Global Average Pooling
- Fully Connected: 256→128→1
- Dropout (0.5) for regularization
```

### NoiseClassifier
```
Architecture:
- Input: 1D PPG signal
- Residual blocks with Conv1D layers
- Block 1: 1→64→64 channels
- Block 2: 64→128→128 channels
- Batch Normalization throughout
- Max Pooling after each residual block
- Global Average Pooling
- Fully Connected: 128→64→2
- Dropout (0.5) for regularization
```

### ImprovedPPGNet (Enhanced Version)
```
Architecture:
- Input: 1D PPG signal
- Initial Conv1D: 1→64 channels
- Three Residual Blocks:
  - Block 1: 64→128 channels
  - Block 2: 128→256 channels
  - Block 3: 256→512 channels
- Attention mechanism on final features
- Global Average Pooling
- Fully Connected: 512→256→128→1
- Dropout (0.3) throughout
```

### Residual Block Design
```
Structure:
- Bottleneck architecture (1x1 → 3x3 → 1x1 convolutions)
- Batch Normalization after each conv
- Dropout (0.3) between layers
- Skip connection with 1x1 conv for dimension matching
- ReLU activation
```

### Dataset Classes
1. **PPGDataset**:
   - Handles raw PPG signal preprocessing
   - Butterworth bandpass filter (0.67-3.0 Hz)
   - Converts signals to PyTorch tensors

2. **NoiseDataset**:
   - Implements 80-20 noise classification split
   - Automatic noise labeling using signal variance
   - Threshold at 80th percentile of noise levels

## Contributing

I welcome contributions! 😊

Elnatan Davidovitch ©
