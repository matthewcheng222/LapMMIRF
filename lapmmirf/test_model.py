import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def test_model(model_path, test_image_pair_loader, device):
    """
    Evaluate a given model on a test dataset.

    Parameters
    ----------
    model_path : str
        Path to the saved model that needs to be evaluated.
    test_image_pair_loader : torch.utils.data.DataLoader
        DataLoader that loads pairs of images for testing.
    device : torch.device
        The device (cpu or gpu) where the data should be loaded.

    Returns
    -------
    mse_avg : float
        The average mean squared error (MSE) over all the samples in the test dataset.
    dice_avg : float
        The average Dice coefficient over all the samples in the test dataset.
    jacobian_avg : float
        The average Jacobian determinant over all the samples in the test dataset.
    avg_time : float
        The average time taken to process each sample in the test dataset.
    """

    # Load the saved model
    model = ImageRegistrationNet().to(device)
    model.load_state_dict(torch.load(model_path))
    
    # Set the model to evaluation mode
    model.eval()

    # Initialize variables to store performance metrics
    mse_sum = 0
    dice_sum = 0
    jacobian_sum = 0
    num_samples = 0
    test_limit = 50
    time_sum = 0
    total_time = 0

    # Ensure no gradients are calculated
    with torch.no_grad():
        while test_limit > 0:
            for fixed_image, moving_image, fixed_image_path, moving_image_path in test_image_pair_loader:
                # Move images to the specified device
                fixed_image, moving_image = fixed_image.to(device), moving_image.to(device)

                # Check if the shapes of the fixed and moving images match
                if fixed_image.shape != moving_image.shape:
                    print("Skipping image pair with mismatched dimensions")
                    continue  # Skip this image pair and move on to the next one

                # Create binary masks for the images
                fixed_image_mask = create_binary_mask(fixed_image.cpu().numpy()[0, 0])
                moving_image_mask = create_binary_mask(moving_image.cpu().numpy()[0, 0])

                start_time = time.time() # Record start time of pair processing

                # Apply the model to the image pair
                displacement_field = model(fixed_image, moving_image)

                # Apply the displacement field to the moving image to align it with the fixed image
                warped_image = nn.functional.grid_sample(moving_image, displacement_field.permute(0, 2, 3, 4, 1), align_corners=True)
                
                # Create a binary mask for the registered image
                registered_image_mask = create_binary_mask(warped_image.detach().cpu().numpy()[0, 0])

                # Calculate performance metrics
                mse = mse_loss(warped_image, fixed_image).item()
                dice = dice_coefficient(fixed_image_mask, registered_image_mask)
                jacobian_det = jacobian_determinant(displacement_field).mean().item()

                elapsed_time = time.time() - start_time # Calculate time taken for processing the pair
                total_time += elapsed_time

                # Accumulate performance metrics
                mse_sum += mse
                dice_sum += dice
                jacobian_sum += jacobian_det
                num_samples += 1
                time_sum += elapsed_time
                test_limit -= 1

                # Save the output images and displacement field
                output_dir = './output/'
                os.makedirs(output_dir, exist_ok=True)

                # Plot and save the fixed image, moving image, registered image and the displacement field
                plt.figure(figsize=(18, 12))
                plt.subplot(2, 2, 1)
                plt.imshow(fixed_image.cpu().numpy()[0, 0, :, :, fixed_image.shape[2] // 2], cmap='gray')
                plt.title('Fixed Image')
                plt.axis('off')
                
                plt.subplot(2, 2, 2)
                plt.imshow(moving_image.cpu().numpy()[0, 0, :, :, moving_image.shape[2] // 2], cmap='gray')
                plt.title('Moving Image')
                plt.axis('off')
                
                plt.subplot(2, 2, 3)
                plt.imshow(registered_image_mask[:, :, registered_image_mask.shape[2] // 2], cmap='gray')
                plt.title('Registered Image')
                plt.axis('off')

                # Quiver plot showing the displacement field
                plt.subplot(2, 2, 4)
                displacement_field_cpu = displacement_field.cpu().numpy()[0]
                X, Y = np.meshgrid(np.arange(0, displacement_field_cpu.shape[2], 20),
                                   np.arange(0, displacement_field_cpu.shape[1], 20))
                U = displacement_field_cpu[0, Y, X, displacement_field_cpu.shape[3] // 2]
                V = displacement_field_cpu[1, Y, X, displacement_field_cpu.shape[3] // 2]
                plt.quiver(X, Y, U, V)
                plt.title('Displacement Field')
                plt.axis('off')

                # Save the final figure as an image
                plt.savefig(os.path.join(output_dir, f'registered_{num_samples}.png'))
                plt.close()

                print("image tested")
                break

    # Calculate average performance metrics
    mse_avg = mse_sum / num_samples
    dice_avg = dice_sum / num_samples
    jacobian_avg = jacobian_sum / num_samples
    avg_time = total_time / num_samples

    # Return the average performance metrics
    return mse_avg, dice_avg, jacobian_avg, avg_time