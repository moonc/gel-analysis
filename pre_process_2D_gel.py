import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# Define the color-to-class mapping
COLOR_TO_CLASS = {
    (0, 0, 0): 0,        # Black -> Background
    (255, 0, 0): 1,      # Red -> Class 1
    (0, 255, 0): 2,      # Green -> Class 2
    (0, 0, 255): 3,      # Blue -> Class 3
    (255, 255, 0): 4,    # Yellow -> Class 4
}

def generate_label_mask(colored_mask_path, color_to_class, output_size=None):
    """
    Convert a colored mask into a label mask with class IDs.

    Args:
        colored_mask_path (str): Path to the colored mask image.
        color_to_class (dict): Mapping of RGB tuples to class IDs.
        output_size (tuple, optional): Desired output size (height, width).

    Returns:
        np.ndarray: A 2D array of class IDs.
    """
    # Load the colored mask
    mask = Image.open(colored_mask_path).convert("RGB")  # Ensure the mask is RGB

    # Resize if output size is specified
    if output_size:
        mask = mask.resize(output_size, Image.NEAREST)

    # Convert mask to a NumPy array
    mask_np = np.array(mask)

    # Initialize an empty label mask
    label_mask = np.zeros((mask_np.shape[0], mask_np.shape[1]), dtype=np.int64)

    # Map each color to its class ID
    for color, class_id in color_to_class.items():
        label_mask[(mask_np == color).all(axis=-1)] = class_id

    return label_mask

def process_colored_masks(input_dir, output_dir, color_to_class, output_size=None):
    """
    Convert all colored masks in a directory to label masks with class IDs.

    Args:
        input_dir (str): Directory containing colored masks.
        output_dir (str): Directory to save the label masks.
        color_to_class (dict): Mapping of RGB tuples to class IDs.
        output_size (tuple, optional): Desired output size (height, width).

    Returns:
        None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mask_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg'))]

    for mask_file in tqdm(mask_files, desc="Processing masks"):
        # Input and output paths
        input_path = os.path.join(input_dir, mask_file)
        output_path = os.path.join(output_dir, os.path.splitext(mask_file)[0] + ".npz")

        # Generate the label mask
        label_mask = generate_label_mask(input_path, color_to_class, output_size)

        # Save as .npz
        np.savez_compressed(output_path, label=label_mask)

    print(f"Processed {len(mask_files)} masks and saved to {output_dir}")


def preprocess_images(input_dir, output_dir, output_size=(224, 224)):
    """
    Preprocess all images in a directory and save them as .npz files.

    Args:
        input_dir (str): Directory containing input images.
        output_dir (str): Directory to save the processed images.
        output_size (tuple): Desired output size (height, width).

    Returns:
        None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg'))]

    for image_file in tqdm(image_files, desc="Processing images"):
        # Input and output paths
        input_path = os.path.join(input_dir, image_file)
        output_path = os.path.join(output_dir, os.path.splitext(image_file)[0] + ".npz")

        # Load the image
        image = Image.open(input_path).convert("L")  # Convert to grayscale if needed
        image = image.resize(output_size, Image.BICUBIC)  # Resize

        # Convert to NumPy array and normalize to [0, 1]
        image_np = np.array(image).astype(np.float32) / 255.0

        # Save as .npz
        np.savez_compressed(output_path, image=image_np)

    print(f"Processed {len(image_files)} images and saved to {output_dir}")

from torch.utils.data import Dataset
import torch

class Synapse2D_Dataset(Dataset):
    """
    Dataset class for 2D image slices stored in .npz files.
    """
    def __init__(self, image_dir, label_dir, list_dir, split, transform=None):
        """
        Args:
            image_dir (str): Directory containing preprocessed images (.npz).
            label_dir (str): Directory containing label masks (.npz).
            list_dir (str): Directory containing split text files (train/test.txt).
            split (str): 'train' or 'test' split.
            transform (callable, optional): Transformations to apply on the samples.
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.split = split
        self.sample_list = open(os.path.join(list_dir, f"{split}.txt")).readlines()

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        # Load the image and label
        sample_name = self.sample_list[idx].strip()
        image_path = os.path.join(self.image_dir, f"{sample_name}.npz")
        label_path = os.path.join(self.label_dir, f"{sample_name}.npz")

        image = np.load(image_path)['image']
        label = np.load(label_path)['label']

        # Apply transformations if defined
        if self.transform:
            sample = {'image': image, 'label': label}
            sample = self.transform(sample)
        else:
            # Convert to tensors
            image = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension
            label = torch.from_numpy(label).long()
            sample = {'image': image, 'label': label}

        return sample
