import os
import random
import glob
import nibabel as nib
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

class ImagePairGenerator(Dataset):
    """
    This class generates a pair of normalized and resampled images at runtime for image registration tasks.

    Parameters:
        fixed_image_dir (str): A string representing the directory path containing fixed images.
        moving_image_dir (str): A string representing the directory path containing moving images.

    Returns:
        None

    Raises:
        None
    """
    def __init__(self, fixed_image_dir, moving_image_dir, target_size=(144, 144, 144)):
        self.fixed_image_dir = fixed_image_dir
        self.moving_image_dir = moving_image_dir
        self.target_size = target_size
        self.fixed_image_paths = sorted(glob.glob(os.path.join(fixed_image_dir, '*.nii.gz')))
        self.moving_image_paths = sorted(glob.glob(os.path.join(moving_image_dir, '*.nii.gz')))

    def __len__(self):
        """
        Returns the number of image pairs in the dataset.

        Parameters:
            None

        Returns:
            int: The number of image pairs in the dataset.

        Raises:
            None
        """
        return len(self.fixed_image_paths)

    def _load_images(self, fixed_image_path, moving_image_path):
        """
        Loads and resamples a fixed and moving image to the desired target size.

        Parameters:
            fixed_image_path (str): A string representing the path of the fixed image.
            moving_image_path (str): A string representing the path of the moving image.

        Returns:
            tuple: A tuple containing the normalized and resampled fixed and moving images as tensors.

        Raises:
            None
        """
        fixed_image = nib.load(fixed_image_path)
        moving_image = nib.load(moving_image_path)

        fixed_image_data = fixed_image.get_fdata()
        moving_image_data = moving_image.get_fdata()

        fixed_resampling_factors = [t / f for t, f in zip(self.target_size, fixed_image_data.shape)]
        moving_resampling_factors = [t / m for t, m in zip(self.target_size, moving_image_data.shape)]

        fixed_image_data = zoom(fixed_image_data, fixed_resampling_factors, order=1)
        moving_image_data = zoom(moving_image_data, moving_resampling_factors, order=1)

        fixed_image_normalized = (fixed_image_data - fixed_image_data.min()) / (fixed_image_data.max() - fixed_image_data.min() + 1e-8)
        moving_image_normalized = (moving_image_data - moving_image_data.min()) / (moving_image_data.max() - moving_image_data.min() + 1e-8)

        fixed_image_tensor = torch.tensor(fixed_image_normalized, dtype=torch.float32).unsqueeze(0)
        moving_image_tensor = torch.tensor(moving_image_normalized, dtype=torch.float32).unsqueeze(0)
        return fixed_image_tensor, moving_image_tensor

    def __getitem__(self, index):
        """
        Returns a pair of normalized and resampled fixed and moving images and their respective paths.

        Parameters:
            index (int): An integer representing the index of the image pair.

        Returns:
            tuple: A tuple containing the normalized and resampled fixed and moving images as tensors,
            and their respective file paths.

        Raises:
            None
        """
        fixed_image_path = self.fixed_image_paths[index]
        moving_image_path = self.moving_image_paths[index]
        fixed_image_tensor, moving_image_tensor = self._load_images(fixed_image_path, moving_image_path)
        return fixed_image_tensor, moving_image_tensor, fixed_image_path, moving_image_path

def visualize_random_images(directory):
    """
    This function loads 5 random images from a directory, plots the middle slice of each image,
    and displays them as subplots in a figure.

    Parameters:
        directory (str): A string representing the directory path where the images are stored.

    Returns:
        None

    Raises:
        None
    """
    # Get all file names in the directory
    all_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    # Check if there are fewer than 5 files in the directory
    if len(all_files) < 5:
        print("There are fewer than 5 files in the directory.")
        return
    
    # Choose 5 random file names from the directory
    random_files = random.sample(all_files, 5)

    # Create a figure with 5 subplots
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    
    # Load each random image and plot the middle slice
    for i, file in enumerate(random_files):
        file_path = os.path.join(directory, file)
        img = nib.load(file_path)
        img_data = img.get_fdata()
        slice_idx = img_data.shape[2] // 2
        axes[i].imshow(img_data[:, :, slice_idx].T, cmap='gray', origin='lower')
        axes[i].axis('off')

    # Show the plot
    plt.show()

def plot_loss_vs_iterations(loss_values):
    """
    Plots the loss values vs. the number of iterations.

    Parameters:
        loss_values (list): A list of loss values.

    Returns:
        None

    Raises:
        None
    """
    # Convert loss values to floats if they are Torch tensors
    loss_values = [float(loss.cpu().detach().numpy()) if isinstance(loss, torch.Tensor) else loss for loss in loss_values]

    # Create a list of iterations
    iterations = list(range(1, len(loss_values) + 1))

    # Plot the loss values vs. the iterations
    plt.figure(figsize=(10, 5))
    plt.plot(iterations, loss_values, marker='o')
    plt.xlabel('Iterations')
    plt.ylabel('Loss Values')
    plt.title('Loss Values vs Iterations')
    plt.grid()
    plt.show()

def plot_dice_vs_iterations(dice_values):
    """
    Plots the Dice coefficient vs. the number of iterations.

    Parameters:
        dice_values (list): A list of Dice coefficient values.

    Returns:
        None

    Raises:
        None
    """
    # Convert Dice coefficient values to floats if they are Torch tensors
    dice_values = [float(dice.cpu().detach().numpy()) if isinstance(dice, torch.Tensor) else dice for dice in dice_values]

    # Create a list of iterations
    iterations = list(range(1, len(dice_values) + 1))

    # Plot the Dice coefficient values vs. the iterations
    plt.figure(figsize=(10, 5))
    plt.plot(iterations, dice_values, marker='o')
    plt.xlabel('Iterations')
    plt.ylabel('Dice Coefficient')
    plt.title('Dice Coefficient vs Iterations')
    plt.grid()
    plt.show()

def plot_time_vs_iterations(time_values):
    """
    Plots the time vs. the number of iterations.

    Parameters:
        time_values (list): A list of time values.

    Returns:
        None

    Raises:
        None
    """
    # Convert time values to floats if they are Torch tensors
    time_values = [float(time.cpu().detach().numpy()) if isinstance(time, torch.Tensor) else time for time in time_values]
    
    # Create a list of iterations
    iterations = list(range(1, len(time_values) + 1))

    # Plot the time values vs. the iterations
    plt.figure(figsize=(10, 5))
    plt.plot(iterations, time_values, marker='o')
    plt.xlabel('Iterations')
    plt.ylabel('Time')
    plt.title('Time vs Iterations')
    plt.grid()
    plt.show()