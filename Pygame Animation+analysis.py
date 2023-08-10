import numpy as np
import matplotlib.pyplot as plt
import librosa
import pandas as pd

# Constants
CHUNK_SIZE = 1024
RATE = 44100

# Load mp3 file and convert to NumPy array
file_path = r"C:\Users\otavi\Downloads\Anavitória - Trevo (Tu) (Audio) ft. Tiago Iorc.mp3"

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
    if len(data) < CHUNK_SIZE:
        continue

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

import pygame
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np

# Initialize pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1080
SCREEN_HEIGHT = 1400

# Set up display and clock
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Frequency Animation')
clock = pygame.time.Clock()

# Function to get the appropriate image
def find_closest_image(radius):
    closest_radius = round(radius)
    image_folder = Path(r"C:\Users\otavi\Documents\images_single_colorNEW")
    closest_image_path = image_folder / f"{closest_radius}.png"
    if not closest_image_path.exists():
        print(f"Image for {closest_radius} not found!")
        return pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))  # return a plain white surface

    img = Image.open(closest_image_path).convert('RGBA')
    background = Image.new('RGBA', img.size, (255, 255, 255, 255))
    background.paste(img, (0, 0), img)
    
    # Convert the PIL Image to a Pygame surface
    return pygame.image.fromstring(background.tobytes(), background.size, background.mode)

# Load audio
audio_path = file_path
audio_clip = pygame.mixer.music.load(audio_path)

# Load frequency-time mappings from CSV
#df = pd.read_csv(r"C:\Users\otavi\Documents\frequency_table.csv", names=["Time", "Frequency"], header=None)

#print(df.head())

times = df["Time"].values
frequencies = df["Frequency"].values

# Start the animation
pygame.mixer.music.play()

running = True
while running:
    screen.fill((255, 255, 255))  # Fill the screen with a white color

    # Get the current position of the audio clip
    current_time = pygame.mixer.music.get_pos() / 1000.0  # Convert from milliseconds to seconds

    # Find the index of the closest time in the CSV to the current time of the audio
    idx = np.searchsorted(times, current_time, side="right") - 1
    
    # Get the corresponding image for that time
    if 0 <= idx < len(times):
        frequency = frequencies[idx]
        image_surface = find_closest_image(frequency)
        screen.blit(image_surface, (0, 0))

    pygame.display.flip()  # Update the screen with the changes
    clock.tick(60)  # Limit the frame rate to 60 FPS

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False




# Cleanup
pygame.mixer.music.stop()
pygame.quit()
