from pathlib import Path

import cv2
import numpy as np
from numpy import typing as npt

from common.data_manipulation import load_image
from common.visualization import compare_images


def add_gaussian_noise(
    image: npt.NDArray, mean: float = 0, std: float = 1
) -> npt.NDArray:
    """
    Adds Gaussian noise to an image.

    Args:
        image (np.array): Original image.
        mean (float): Mean of the Gaussian noise.
        std (float): Standard deviation of the Gaussian noise.

    Returns:
        np.array: Noisy image.
    """
    gaussian = np.random.normal(mean, std, image.shape)
    noisy_image = image + gaussian
    noisy_image = np.clip(
        noisy_image, 0, 255
    )  # Ensure values are within [0, 255]
    return noisy_image


def add_speckle_noise(
    image: npt.NDArray, intensity: float = 0.05
) -> npt.NDArray:
    """
    Adds speckle noise to an image.

    Args:
        image (npt.NDArray): Original image.
        intensity (float): Intensity factor for the noise.

    Returns:
        npt.NDArray: Noisy image.
    """
    noise = np.random.normal(0, 1, image.shape)
    noisy_image = image + image * noise * intensity
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image


def add_salt_and_pepper_noise(
    image: npt.NDArray, salt_prob: float = 0.05, pepper_prob: float = 0.05
) -> npt.NDArray:
    """
    Adds salt and pepper noise to an image.

    Args:
        image (np.array): Original image.
        salt_prob (float): Probability of adding salt (white) noise.
        pepper_prob (float): Probability of adding pepper (black) noise.

    Returns:
        np.array: Noisy image.
    """
    noisy_image = np.copy(image)
    salt = np.random.rand(*image.shape) < salt_prob
    pepper = np.random.rand(*image.shape) < pepper_prob

    noisy_image[salt] = 255
    noisy_image[pepper] = 0
    return noisy_image


def add_poisson_noise(
    image: npt.NDArray, intensity: float = 1.0
) -> npt.NDArray:
    """
    Adds Poisson (Shot) noise to an image with adjustable intensity.

    Args:
        image (npt.NDArray): Original image.
        intensity (float): Intensity factor for the noise.

    Returns:
        npt.NDArray: Noisy image.
    """
    scaled_image = image * intensity

    # Calculate the maximum value to scale the image for Poisson distribution
    unique_pixels = len(np.unique(image))
    max_val = 2 ** np.ceil(np.log2(unique_pixels))
    scaled_image = np.clip(scaled_image, 0, max_val - 1)

    noisy_image = np.random.poisson(scaled_image) / intensity

    return noisy_image


def add_motion_blur(
    image: npt.NDArray, size: int = 15, angle: int = 0
) -> npt.NDArray:
    """
    Applies motion blur to an image.

    Args:
        image (npt.NDArray): Original image.
        size (int): Size of the motion blur kernel.
        angle (int): Angle of the motion blur (in degrees).

    Returns:
        npt.NDArray: Blurred image.
    """
    M = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1)
    kernel = np.diag(np.ones(size))
    kernel = cv2.warpAffine(kernel, M, (size, size))

    kernel = kernel / size
    blurred_image = cv2.filter2D(image, -1, kernel)

    return blurred_image


def add_gaussian_blur(image: npt.NDArray, sigma: float = 2.0) -> npt.NDArray:
    """
    Applies out-of-focus blur to an image to simulate focus issues during image acquisition.

    Args:
        image (npt.NDArray): Original image.
        sigma (float): Standard deviation of the Gaussian kernel for blur.

    Returns:
        npt.NDArray: Blurred image.
    """
    return cv2.GaussianBlur(image, (0, 0), sigma)


def add_beam_hardening_artifact(
    image: npt.NDArray,
    intensity: float = 0.2,
    lower_bound: float = 1.0,
    upper_bound: float = 1.2,
) -> npt.NDArray:
    """
    Simulates beam hardening artifact in X-ray images.

    Args:
        image (npt.NDArray): Original image.
        intensity (float): Intensity of the beam hardening artifact.
        lower_bound (float): Lower bound for artifact scaling.
        upper_bound (float): Upper bound for artifact scaling.

    Returns:
        npt.NDArray: Image with beam hardening artifact.
    """
    artifact = (
        np.random.uniform(lower_bound, upper_bound, image.shape) * intensity
    )
    return (image * artifact).clip(0, 255).astype(np.uint8)


def test_transformations(image_path: Path) -> None:
    original_image = load_image(image_path)

    transformations = [
        {
            "function": add_gaussian_noise,
            "params": {"mean": 0, "std": 10},
            "title": "Gaussian Noise",
        },
        {
            "function": add_speckle_noise,
            "params": {"intensity": 0.05},
            "title": "Speckle Noise",
        },
        {
            "function": add_salt_and_pepper_noise,
            "params": {"salt_prob": 0.02, "pepper_prob": 0.02},
            "title": "Salt and Pepper Noise",
        },
        {"function": add_poisson_noise, "params": {}, "title": "Poisson Noise"},
        {
            "function": add_gaussian_blur,
            "params": {"sigma": 5},
            "title": "Gaussian blur",
        },
        {
            "function": add_motion_blur,
            "params": {"size": 15, "angle": 45},
            "title": "Motion Blur",
        },
        {
            "function": add_beam_hardening_artifact,
            "params": {"intensity": 0.2},
            "title": "Beam Hardening Artifact",
        },
    ]

    for transformation in transformations:
        transformed_image = transformation["function"](
            original_image, **transformation["params"]
        )
        compare_images(
            original=original_image,
            transformed=transformed_image,
            original_title="Original Image",
            transformed_title=transformation["title"],
        )
