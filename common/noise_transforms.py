from pathlib import Path
from typing import Any, Dict

import cv2
from PIL import Image
import numpy as np
from numpy import typing as npt

from common.data_manipulation import load_image
from common.visualization import compare_images


class NoiseTransformHandler:
    @staticmethod
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
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return noisy_image

    @staticmethod
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
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return noisy_image

    @staticmethod
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

    @staticmethod
    def add_poisson_noise(
        image: npt.NDArray, intensity: float = 1.0
    ) -> float:
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

    @staticmethod
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

        kernel /= size
        blurred_image = cv2.filter2D(image, -1, kernel)

        return blurred_image

    @staticmethod
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

    @staticmethod
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

    def apply_noise_transform(
        self, image: npt.NDArray, transform_type: str, params: dict | None = None
    ) -> Image.Image:
        transform_method = getattr(self, f"add_{transform_type}")

        if transform_method and callable(transform_method):
            if params is None:
                return Image.fromarray(transform_method(image))
            else:
                return Image.fromarray(transform_method(image, **params))
        else:
            return Image.fromarray(image)


def test_noise_transforms(
    image_path: Path, noise_transform_config: Dict[str, Dict[str, Any]]
) -> None:
    original_image = load_image(image_path)
    noise_transform_handler = NoiseTransformHandler()
    noise_types = list(noise_transform_config.keys())

    for noise_type in noise_types:
        params = noise_transform_config[noise_type]
        noised_image = noise_transform_handler.apply_noise_transform(
            np.array(original_image).copy(), noise_type, params
        )
        compare_images(
            images=[original_image, noised_image],
            titles=["Original image", "Noised image"],
            images_per_column=2
        )
