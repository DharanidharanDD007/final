import numpy as np

print("Generating synthetic spectrogram dataset for testing...")
num_samples = 1000 # 500 Authentic, 500 Deepfake

# Create dummy 64x64 frequency spectrograms
X = np.random.rand(num_samples, 64, 64, 1)
# Create labels: 0 for Authentic, 1 for Deepfake
y = np.array([0]*500 + [1]*500)

np.save('X_spectrograms.npy', X)
np.save('y_labels.npy', y)

print("Success! Saved X_spectrograms.npy and y_labels.npy")