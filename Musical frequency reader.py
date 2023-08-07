#pip install numpy
#pip install matplotlib
#pip install librosa
#pip install pandas




import numpy as np
import matplotlib.pyplot as plt
import librosa
import pandas as pd

# Constants
CHUNK_SIZE = 1024
RATE = 44100

# Load mp3 file and convert to NumPy array
file_path = r"C:\Users\otavi\Downloads\Chopin - Heroic Polonaise (Op. 53 in A Flat Major).mp3"

# Load audio file
y, sr = librosa.load(file_path, sr=RATE, mono=True)

# Convert to NumPy array
audio_data = y.astype(np.float32)

# Create time and frequency axes
freq_axis = np.linspace(0, RATE/2, CHUNK_SIZE//2)

# Define autocorrelation function
def autocorrelate(signal):
    correlation = np.correlate(signal, signal, mode="full")
    return correlation[len(correlation)//2:]

# Loop through audio data in chunks
peak_freqs = [] # initialize list to store peak frequencies
for i in range(0, len(audio_data), CHUNK_SIZE):
    # Extract chunk of audio data
    data = audio_data[i:i+CHUNK_SIZE]
    
    # Apply window function to reduce spectral leakage
    window = np.hamming(len(data))
    windowed_data = data * window
    
    # Compute autocorrelation function
    autocorr_data = autocorrelate(windowed_data)
    
    # Find peak in autocorrelation function
    peak_idx = np.argmax(autocorr_data[100:CHUNK_SIZE//2]) + 100
    peak_freq = RATE / peak_idx
    
    # Append peak frequency to list
    peak_freqs.append(peak_freq)

# Create time axis based on length of peak_freqs
time_axis = np.arange(len(peak_freqs)) * (CHUNK_SIZE / RATE)

# Plot peak frequencies against time axis
plt.plot(time_axis, peak_freqs)
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.show()

#Save file as text
table = np.column_stack((time_axis, peak_freqs))

# Save table to CSV file
np.savetxt("frequency_table.csv", table, delimiter=",")
df = pd.read_csv("frequency_table.csv", names=["Time", "Frequency"])