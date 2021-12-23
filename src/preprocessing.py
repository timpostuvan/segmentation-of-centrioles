import torch
from torchvision import transforms
import numpy as np
from scipy import signal

import constants
import utils

class RandomGaussianNoise(object):
    "Adds random gaussian noise to the image"
    
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        noise = torch.normal(mean=self.mean, std=self.std, size=image.shape)
        return (image + noise)


def standardize_patch(image):
    """
    Standardize a tensor patch

    Args:
        image : the image to be standardized

    Returns:
        image: the standardized image
    """

    image_mean = torch.mean(image, dim=0)
    image_std = torch.std(image, dim=0) + 1e-10
    normalization_transform = transforms.Normalize(mean=image_mean, std=image_std)
    image = normalization_transform(image)
    return image


def normalize(image):
    """
    Normalize each channel separately to [0, 1] interval

    Args:
        image : the image to be normalized

    Returns:
        normalized_image: the normalized image
    """

    normalized_image = np.zeros(image.shape, dtype=np.float)
    for i in range(image.shape[2]):
        min_val = np.min(image[:, :, i])
        max_val = np.max(image[:, :, i])
        range_val = max_val - min_val + 1e-10

        normalized_image[:, :, i] = np.array((image[:, :, i] - min_val) 
                                            / float(range_val), dtype=np.float)
    
    return normalized_image


def log_normalize(image):
    """
    Normalize log of each channel separately to [0, 1] interval

    Args:
        image : the image to be log normalized

    Returns:
        normalized_image: the log normalize image
    """

    normalized_image = np.zeros(image.shape, dtype=np.float)
    for i in range(image.shape[2]):
        min_val = np.min(np.log(1 + image[:, :, i]))
        max_val = np.max(np.log(1 + image[:, :, i]))
        range_val = max_val - min_val + 1e-10

        normalized_image[:, :, i] = np.array((np.log(1 + image[:, :, i]) - min_val) 
                                            / float(range_val), dtype=np.float)

    return normalized_image


def thresholded_normalize(image, percentile=95):
    """
    Normalize image data that are above the percentile argument

    Args:
        image : the image to be normalized
        percentile (int, optional): threshold quantity. Defaults to 95.

    Returns:
        normalized_image: the normalized image
    """
    normalized_image = np.zeros(image.shape, dtype=np.float)
    for i in range(image.shape[2]):
        q = np.percentile(image[:, :, i], q=percentile)
        max_val = np.max(image[:, :, i])
        range_val = max_val - q + 1e-10

        normalized_image[image[:, :, i] < q, i] = -1
        normalized_image[image[:, :, i] >= q, i] = np.array((image[image[:, :, i] >= q, i] - q) 
                                                           / range_val, dtype=np.float)
    
    return normalized_image


def standardize(image):
    """
    Standardize each channel separately

    Args:
        image : the image to be normalized

    Returns:
        standardized_image: the standardized image
    """

    standardized_image = np.zeros(image.shape, dtype=np.float)
    for i in range(image.shape[2]):
        mean = np.mean(image[:, :, i])
        std = np.std(image[:, :, i]) + 1e-10

        standardized_image[:, :, i] = np.array((image[:, :, i] - mean) / float(std), dtype=np.float)
    
    return standardized_image


def nonmaxima_suppression_box(image, size=3):
    """
    Perform the nonmaxima_supression function

    Args:
        image : [description]
        size (int, optional): kernel size. Defaults to 3.

    Returns:
        image: the image with non-maxima suppressed
    """
    domain = np.ones((size, size))
    max_val = signal.order_filter(image, domain, 8)
    image = np.where(image == max_val, image, 0)
    return image