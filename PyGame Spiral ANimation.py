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
audio_path = r"C:\Users\otavi\Downloads\Queen Bohemian Rhapsody.mp3"
audio_clip = pygame.mixer.music.load(audio_path)

# Load frequency-time mappings from CSV
df = pd.read_csv(r"C:\Users\otavi\Documents\frequency_table.csv", names=["Time", "Frequency"], header=None)

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


frame_number = 0

while running:  # This is your game loop or animation loop
    # Your animation update logic and drawing here...

    pygame.display.flip()

    # Save the current frame as an image
    pygame.image.save(screen, f"frame_{frame_number}.png")
    frame_number += 1


# Cleanup
pygame.mixer.music.stop()
pygame.quit()
