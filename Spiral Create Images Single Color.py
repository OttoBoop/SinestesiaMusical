import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pathlib import Path


def spiral(phi, a=16.5, rotations=5, r_max=1046.5 / 2):
    b = np.log(r_max / a) / (2 * np.pi * rotations)
    return a * np.exp(b * phi)
r_max=(1046.5/2)
def create_colored_spiral_image(radius, output_dir):
    phi = np.linspace(0, 10 * np.pi, 1000)
    r = spiral(phi)
    colored_r = r[r <= radius]
    colored_phi = phi[:len(colored_r)]
    if len(colored_phi) == 0:
    # For now, just return and skip this radius.
        return

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    
    # 1. Remove the loop for the multi-colored spiral
    # Create a black line for the entire spiral
    ax.plot(phi, r, 'k', linewidth=0.5, alpha=0.3)

    # 2. Compute the hue for the chosen frequency (angle)
    final_angle = colored_phi[-1] % (2 * np.pi)
    hue = final_angle / (2 * np.pi)

    # 3. Apply this single color to the entire spiral
    color = colors.hsv_to_rgb((hue, 1, 1))
    ax.fill_between(colored_phi, 0, colored_r, color=color)
    
    # Styling and Saving remains the same
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.grid(False)
    plt.axis('off')
    plt.savefig(output_dir / f"{radius}.png", dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig)

image_folder = Path("images_single_color")
image_folder.mkdir(exist_ok=True)

for radius in range(16, 600):
    create_colored_spiral_image(radius, image_folder)
    print(radius)
