import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pathlib import Path

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

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    #plt.grid(False)
    #plt.axis('off')
    # Create a black line for the entire spiral
    ax.plot(phi, r, 'k', linewidth=0.5, alpha=0.3)

    # Plot the colored spiral
    for i in range(len(colored_r) - 1):
        angle = colored_phi[i] % (2 * np.pi)
        color = angle / (2 * np.pi)
        ax.fill_between(colored_phi[i:i+2], 0, colored_r[i:i+2], color=colors.hsv_to_rgb((color, 1, 1)))

    #ax.set_rmax(1046.5/2)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.grid(False)
    plt.axis('off')
    plt.savefig(output_dir / f"{radius}.png", dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig)

image_folder = Path("images")
image_folder.mkdir(exist_ok=True)

for radius in range(16, 600):
    create_colored_spiral_image(radius, image_folder)
    print(radius)
