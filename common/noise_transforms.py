import cv2
import numpy as np
from numpy import typing as npt


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
    return np.clip(noisy_image, 0, 255).astype(np.uint8)


def add_speckle_noise(image: npt.NDArray, intensity: float) -> npt.NDArray:
    """
    Adds speckle noise to an image.

    Args:
        image (npt.NDArray): Original image.
        intensity (float): Intensity of the applied noise.

    Returns:
        npt.NDArray: Noisy image.
    """
    noise = np.random.normal(0, intensity, image.shape)
    noisy_image = image + image * noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)


def add_salt_and_pepper_noise(
    image: npt.NDArray,
    salt_prob: float = 0.05,
    pepper_prob: float = 0.05,
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
    salt = np.random.rand(*image.shape) < salt_prob
    pepper = np.random.rand(*image.shape) < pepper_prob

    image[salt] = 255
    image[pepper] = 0
    return image


def add_poisson_noise(image: npt.NDArray) -> npt.NDArray:
    """
    Adds Poisson (Shot) noise to an image with adjustable intensity.

    Args:
        image (npt.NDArray): Original image.

    Returns:
        npt.NDArray: Noisy image.
    """
    noisy_image = np.random.poisson(image).astype(image.dtype)
    return np.clip(noisy_image, 0, 255).astype(np.uint8)


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
    matrix = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1)
    kernel = np.diag(np.ones(size))
    kernel = cv2.warpAffine(kernel, matrix, (size, size))

    kernel /= size # type: ignore
    blurred_image = cv2.filter2D(image, -1, kernel)

    return blurred_image


def add_gaussian_blur(
    image: npt.NDArray, sigma: float = 2.0
) -> npt.NDArray:
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
