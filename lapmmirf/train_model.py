import sys
import time
import torch
import torch.nn as nn

def train_model(model, image_pair_loader, optimizer, scheduler, device, num_epochs):
    """
    Train a given model for a certain number of epochs.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained.
    image_pair_loader : torch.utils.data.DataLoader
        The DataLoader which loads pairs of images for training.
    optimizer : torch.optim.Optimizer
        The optimizer to be used for model training.
    scheduler : torch.optim.lr_scheduler._LRScheduler
        The learning rate scheduler.
    device : torch.device
        The device (cpu or gpu) where the data should be loaded.
    num_epochs : int
        The number of epochs for which the model should be trained.

    Returns
    -------
    model : torch.nn.Module
        The trained model.
    training_accuracy_avg : float
        The average training accuracy over all epochs.
    dice_avg : float
        The average Dice coefficient over all epochs.
    jacobian_avg : float
        The average Jacobian determinant over all epochs.
    loss_values : list
        The list of loss values at each step.
    dice_values : list
        The list of Dice coefficients at each step.
    time_values : list
        The list of time taken at each step.
    total_time : float
        The total time taken for training.
    """
    
    # Initialize empty lists to store loss, dice and time values at each step
    loss_values = []
    dice_values = []
    time_values = []

    # Initialize counters for total time, training accuracy, dice and jacobian
    total_time = 0
    training_accuracy_sum = 0
    dice_sum = 0
    jacobian_sum = 0

    # Initialize counter for valid epochs
    valid_epochs = 0

    # Number of levels for Laplacian pyramid loss
    levels = 5

    # Start training loop for each epoch
    for epoch in range(num_epochs):
        # Initialize counters for each epoch
        image_step = 1
        training_accuracy_epoch = 0
        dice_epoch = 0
        valid_steps = 0

        epoch_start_time = time.time() # Record start time of epoch

        # Load image pairs from the dataloader
        for fixed_image, moving_image, fixed_image_path, moving_image_path in image_pair_loader:
            pair_start_time = time.time() # Record start time of pair processing

            # Move images to the specified device
            fixed_image, moving_image = fixed_image.to(device), moving_image.to(device)

            # Check if the shapes of the fixed and moving images match
            if fixed_image.shape != moving_image.shape:
                print("Skipping image pair with mismatched dimensions")
                continue  # Skip this image pair and move on to the next one
            
            valid_epochs += 1
            valid_steps += 1

            # Create binary masks for the images
            fixed_image_mask = create_binary_mask(fixed_image.cpu().numpy()[0, 0])
            moving_image_mask = create_binary_mask(moving_image.cpu().numpy()[0, 0])

            # Clear gradients from the previous iteration
            optimizer.zero_grad()

            # Run the model and compute the displacement field
            displacement_field = model(fixed_image, moving_image)

            # Reshape and normalize the displacement field to create a grid
            grid = displacement_field.permute(0, 2, 3, 4, 1)
            grid = torch.stack([
                2 * grid[:, :, :, :, 0] / (grid.shape[1] - 1),  # Normalize x coordinates
                2 * grid[:, :, :, :, 1] / (grid.shape[2] - 1),  # Normalize y coordinates
                2 * grid[:, :, :, :, 2] / (grid.shape[3] - 1)   # Normalize z coordinates
            ], dim=-1)

            # Apply the displacement field to the moving image to align it with the fixed image
            warped_image = nn.functional.grid_sample(moving_image, grid, align_corners=True)

            # Create a binary mask for the registered image
            registered_image_mask = create_binary_mask(warped_image.detach().cpu().numpy()[0, 0])

            pair_end_time = time.time() # Record end time of pair processing
            pair_time = pair_end_time - pair_start_time # Calculate time taken to process the pair

            # Calculate training accuracy using Mean Squared Error (MSE) loss
            training_accuracy = mse_loss(warped_image, fixed_image).item()

            # Calculate Dice coefficient to measure overlap between fixed_image_mask and registered_image_mask
            dice = dice_coefficient(fixed_image_mask, registered_image_mask)

            # Calculate the determinant of the Jacobian of the displacement field
            jacobian_det = jacobian_determinant(displacement_field)

            # Calculate the Laplacian pyramid loss
            loss = laplacian_pyramid_loss(warped_image, fixed_image, levels, device)
            
            # Perform backpropagation
            loss.backward()
            
            # Update the weights
            optimizer.step()

            # Append the loss, dice and time values to the respective lists
            loss_values.append(loss)
            dice_values.append(dice)
            time_values.append(pair_time)

            # Update the training accuracy, dice, and jacobian sums
            training_accuracy_sum += training_accuracy
            training_accuracy_epoch += training_accuracy
            dice_sum += dice
            dice_epoch += dice
            jacobian_sum += jacobian_det.mean().item()

            # Update the learning rate based on the training accuracy
            scheduler.step(training_accuracy)

            epoch_time = time.time() - epoch_start_time # Calculate time taken for the epoch
            total_time += epoch_time # Update total time

            # Print training loss and dice coefficient for the current step
            sys.stdout.write("\r" + 'step "{0}" -> training loss "{1:.4f}" - dice coefficient "{2:.4f}"'.format(image_step, training_accuracy, dice))
            sys.stdout.flush()
            image_step += 1
        
        # Calculate average training accuracy and dice coefficient for the epoch
        training_accuracy_epoch = training_accuracy_epoch / valid_steps
        dice_epoch = dice_epoch / valid_steps

        # Print epoch summary
        print(f"\nEpoch [{epoch+1}/{num_epochs}], Training Accuracy (MSE): {training_accuracy_epoch:.4f}, Dice Coefficient: {dice_epoch:.4f}, Time elapsed: {epoch_time:.2f} seconds")
    
    # Calculate average training accuracy, dice coefficient, and jacobian determinant for all epochs
    training_accuracy_avg = training_accuracy_sum / valid_epochs
    dice_avg = dice_sum / valid_epochs
    jacobian_avg = jacobian_sum / valid_epochs

    # Return the trained model and the computed metrics
    return model, training_accuracy_avg, dice_avg, jacobian_avg, loss_values, dice_values, time_values, total_time