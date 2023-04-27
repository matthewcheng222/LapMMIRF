import numpy as np
from skimage.filters import threshold_otsu, threshold_local
from skimage.measure import label
import torch
import torch.nn as nn

def mse_loss(registered_image, fixed_image):
    """
    Computes the mean squared error loss between a registered image and a fixed image.

    Parameters:
        registered_image (tensor): A tensor representing the registered image.
        fixed_image (tensor): A tensor representing the fixed image.

    Returns:
        tensor: A tensor representing the mean squared error loss.

    Raises:
        None
    """
    return torch.mean((registered_image - fixed_image) ** 2)

def jacobian_determinant(displacement_field):
    """
    Computes the Jacobian determinant of the displacement field.

    Parameters:
        displacement_field (tensor): A tensor representing the displacement field.

    Returns:
        tensor: A tensor representing the Jacobian determinant of the displacement field.

    Raises:
        None
    """
    # Calculate the gradients of the displacement field
    grad_x = torch.gradient(displacement_field[:, 0, :, :, :], dim=(1, 2, 3))
    grad_y = torch.gradient(displacement_field[:, 1, :, :, :], dim=(1, 2, 3))
    grad_z = torch.gradient(displacement_field[:, 2, :, :, :], dim=(1, 2, 3))

    gradients = torch.stack([
        grad_x[0], grad_x[1], grad_x[2],
        grad_y[0], grad_y[1], grad_y[2],
        grad_z[0], grad_z[1], grad_z[2]
    ], dim=1)

    # Create the Jacobian matrix (3x3) for each voxel
    jacobian = gradients.permute(0, 2, 3, 4, 1).view(-1, 3, 3)

    # Compute the determinant of the Jacobian matrix for each voxel
    jacobian_det = torch.det(jacobian)
    jacobian_det = jacobian_det.view(displacement_field.shape[0], displacement_field.shape[2], displacement_field.shape[3], displacement_field.shape[4])

    return jacobian_det

def dice_coefficient(true_mask, pred_mask, eps=1e-8):
    """
    Computes the Dice coefficient between a true mask and a predicted mask.

    Parameters:
        true_mask (ndarray): A numpy array representing the true mask.
        pred_mask (ndarray): A numpy array representing the predicted mask.
        eps (float): A small number to avoid division by zero.

    Returns:
        float: The Dice coefficient between the true and predicted masks.

    Raises:
        None
    """
    true_mask = torch.from_numpy(true_mask).float()
    pred_mask = torch.from_numpy(pred_mask).float()

    intersection = torch.sum(true_mask * pred_mask)
    union = torch.sum(true_mask) + torch.sum(pred_mask)

    if torch.all(true_mask == 0) and torch.all(pred_mask == 0):
        return 1.0

    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.item()

def create_binary_mask(image, threshold=None, method='otsu'):
    """
    Creates a binary mask from an input image using thresholding.

    Parameters:
        image (ndarray): A numpy array representing the input image.
        threshold (float): A threshold value for binarizing the image.
        method (str): A string indicating the method to use for thresholding.
            Valid options are ['otsu', 'percentile', 'adaptive'].

    Returns:
        ndarray: A numpy array representing the binary mask.

    Raises:
        ValueError: If an invalid thresholding method is specified.
    """
    if threshold is None:
        if method == 'otsu':
            threshold = threshold_otsu(image)
        elif method == 'percentile':
            threshold = np.percentile(image, 90)
        elif method == 'adaptive':
            block_size = 35
            threshold = threshold_local(image, block_size, offset=10)
        else:
            raise ValueError("Invalid thresholding method. Choose from ['otsu', 'percentile', 'adaptive']")

    binary_mask = image > threshold
    label_image = label(binary_mask)
    label_counts = np.bincount(label_image.flat)[1:]

    if len(label_counts) == 0:
        # No connected regions found, return an empty mask
        return np.zeros_like(image, dtype=np.float32)

    largest_label = np.argmax(label_counts) + 1
    binary_mask = label_image == largest_label
    return binary_mask.astype(np.float32)

def laplacian_pyramid_loss(prediction, target, levels, device):
    """
    Calculates the Laplacian pyramid loss between the predicted and target images.

    Parameters:
        prediction (torch.Tensor): The predicted image tensor.
        target (torch.Tensor): The target image tensor.
        levels (int): The number of levels in the Laplacian pyramid.
        device (torch.device): The device on which to perform the calculations.

    Returns:
        torch.Tensor: The Laplacian pyramid loss between the predicted and target images.

    Raises:
        None
    """
    # Create a Laplacian pyramid with the specified number of levels
    lp = LaplacianPyramid3D(levels)
    lp = lp.to(device)
    
    # Compute the Laplacian pyramids for the predicted and target images
    pred_pyramid = lp(prediction)
    target_pyramid = lp(target)

    # Compute the MSE loss between the Laplacian pyramid levels
    loss = 0
    for pred_level, target_level in zip(pred_pyramid, target_pyramid):
        loss += nn.functional.mse_loss(pred_level, target_level)

    return loss