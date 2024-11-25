import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def visualize_mask(image_path, mask_path, color_to_class, class_to_color=None):
    """
    Visualize an image and its corresponding label mask.

    Args:
        image_path (str): Path to the preprocessed image (.npz).
        mask_path (str): Path to the label mask (.npz).
        color_to_class (dict): Mapping of RGB colors to class IDs (used for class verification).
        class_to_color (dict, optional): Mapping of class IDs to RGB colors for visualization.

    Returns:
        None
    """
    # Load the image and mask
    image = np.load(image_path)['image']
    label_mask = np.load(mask_path)['label']

    # Ensure the color mapping is correct
    if class_to_color is None:
        class_to_color = {v: k for k, v in color_to_class.items()}

    # Create a color map for visualization
    num_classes = len(class_to_color)
    colors = [np.array(class_to_color[i]) / 255.0 if i in class_to_color else [0, 0, 0] for i in range(num_classes)]
    cmap = ListedColormap(colors)

    # Visualize the image and mask
    plt.figure(figsize=(10, 5))

    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Input Image")
    plt.axis("off")

    # Label mask
    plt.subplot(1, 2, 2)
    plt.imshow(label_mask, cmap=cmap, interpolation='nearest')
    plt.title("Label Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def verify_masks(image_dir, mask_dir, num_samples=1):
    """
    Verify a few samples from the dataset by visualizing the image and its mask.

    Args:
        image_dir (str): Directory containing preprocessed images (.npz).
        mask_dir (str): Directory containing label masks (.npz).
        num_samples (int): Number of samples to visualize.

    Returns:
        None
    """
    # List of images and masks
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.npz')])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.npz')])

    # Ensure both directories match
    assert len(image_files) == len(mask_files), "Mismatch between image and mask files."

    # Visualize a few samples
    for i in range(min(num_samples, len(image_files))):
        image_path = os.path.join(image_dir, image_files[i])
        mask_path = os.path.join(mask_dir, mask_files[i])

        print(f"Visualizing {image_files[i]} and {mask_files[i]}")
        visualize_mask(image_path, mask_path, COLOR_TO_CLASS)

# Define the COLOR_TO_CLASS and CLASS_TO_COLOR mappings
COLOR_TO_CLASS = {
    (0, 0, 0): 0,        # Black -> Background
    (255, 0, 0): 1,      # Red -> Class 1
    (0, 255, 0): 2,      # Green -> Class 2
    (0, 0, 255): 3,      # Blue -> Class 3
    (255, 255, 0): 4,    # Yellow -> Class 4
}
CLASS_TO_COLOR = {v: k for k, v in COLOR_TO_CLASS.items()}  # Invert for visualization

# Verify masks
verify_masks(
    image_dir="dataset/preprocessed_images",
    mask_dir="dataset/label_masks",
    num_samples=5  # Number of samples to visualize
)
