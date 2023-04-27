import torch
import torch.nn as nn
import torch.nn.functional as F

class LaplacianPyramid3D(nn.Module):
    """
    This class implements a 3D Laplacian pyramid for image processing.

    Parameters:
        levels (int): An integer representing the number of pyramid levels.

    Returns:
        None

    Raises:
        None
    """
    def __init__(self, levels):
        super(LaplacianPyramid3D, self).__init__()
        self.levels = levels
        self.gauss_pyramid = nn.ModuleList([nn.AvgPool3d(2, 2) for _ in range(levels)])

    def forward(self, img):
        """
        Builds a 3D Laplacian pyramid from an input tensor.

        Parameters:
            img (tensor): A tensor representing the input image.

        Returns:
            list: A list of tensors representing the levels of the Laplacian pyramid.

        Raises:
            None
        """
        pyramid = []
        current = img
        for level in range(self.levels):
            downsampled = self.gauss_pyramid[level](current)
            upsampled = nn.functional.interpolate(downsampled, size=current.shape[2:], mode='trilinear', align_corners=True)

            # Pad the upsampled tensor if needed
            pad_dims = [0, 0, 0, 0, 0, 0]  # padding for (left, right, top, bottom, front, back)
            for i in range(2, 5):
                if upsampled.size(i) < current.size(i):
                    pad_dims[2 * (5 - i)] = 1

            if sum(pad_dims) > 0:
                upsampled = nn.functional.pad(upsampled, pad_dims)

            laplacian = current - upsampled
            pyramid.append(laplacian)
            current = downsampled
        pyramid.append(current)
        return pyramid
    
class UNet(nn.Module):
    """
    This class implements a 3D U-Net for image segmentation.

    Parameters:
        in_channels (int): An integer representing the number of input channels.
        out_channels (int): An integer representing the number of output channels.

    Returns:
        None

    Raises:
        None
    """ 
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.middle = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Conv3d(128 + 64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        """
        Implements a forward pass through the U-Net architecture.

        Parameters:
            x (tensor): A tensor representing the input image.

        Returns:
            tensor: A tensor representing the output segmentation mask.

        Raises:
            None
        """
        x1 = self.encoder(x)
        x2 = F.max_pool3d(x1, kernel_size=2, stride=2)
        x3 = self.middle(x2)
        x4 = F.interpolate(x3, scale_factor=2, mode='trilinear', align_corners=True)
        x5 = torch.cat([x4, x1], dim=1)
        x6 = self.decoder(x5)
        return x6
    
class ImageRegistrationNet(nn.Module):
    """
    This class implements a deep learning model for image registration.

    Parameters:
        None

    Returns:
        None

    Raises:
        None
    """
    def __init__(self):
        super(ImageRegistrationNet, self).__init__()
        self.unet = UNet(in_channels=2, out_channels=3)

    def forward(self, fixed_image, moving_image):
        """
        Implements a forward pass through the image registration network.

        Parameters:
            fixed_image (tensor): A tensor representing the fixed image.
            moving_image (tensor): A tensor representing the moving image.

        Returns:
            tensor: A tensor representing the displacement field.

        Raises:
            None
        """
        x = torch.cat((fixed_image, moving_image), dim=1)
        displacement_field = self.unet(x)
        return displacement_field