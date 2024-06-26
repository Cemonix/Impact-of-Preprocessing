from typing import Tuple, cast
import numpy as np
from numpy import typing as npt
from skimage import exposure, filters, restoration
from skimage.util import img_as_float
from scipy.ndimage import generic_filter, median_filter
import numba


def apply_histogram_equalization(image: npt.NDArray) -> npt.NDArray:
    """
    Perform histogram equalization on a grayscale image.

    This function applies histogram equalization to enhance the contrast of a
    grayscale image. It computes the histogram of the image, calculates the
    cumulative distribution function (CDF), and uses it to transform the pixel
    values. The transformed image is returned as a new NumPy array with enhanced
    contrast.

    Args:
        image (numpy.ndarray): A 2D NumPy array representing the grayscale image
                             to be processed. It should be an 8-bit image with
                             values ranging from 0 to 255.

    Returns:
        numpy.ndarray: A 2D NumPy array representing the histogram-equalized image.
    """
    histogram, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
    cdf = histogram.cumsum()
    cdf_min = cdf.min()

    transformed_img: npt.NDArray = np.round(
        (cdf[image] - cdf_min) / (image.shape[0] * image.shape[1] - cdf_min) * 255
    )
    return transformed_img.astype("uint8")


def apply_clahe(
    image: npt.NDArray, clip_limit: float = 0.01, nbins: int = 256
) -> npt.NDArray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to the input image.

    Parameters:
        image (numpy.ndarray): The input image.
        clip_limit (float, optional): Threshold for contrast limiting. Default is 0.01.
        nbins (int, optional): Number of histogram bins. Default is 256.

    Returns:
        numpy.ndarray: The processed image after applying CLAHE.

    """
    processed_image = exposure.equalize_adapthist(
        image, clip_limit=clip_limit, nbins=nbins
    )
    return (processed_image * 255).astype(np.uint8)


def apply_gamma_correction(image: npt.NDArray, gamma: int = 1) -> npt.NDArray:
    processed_image = exposure.adjust_gamma(image, gamma)
    return (processed_image * 255).astype(np.uint8)


def apply_unsharp_masking(
    image: npt.NDArray, radius: float = 0.0, amount: float = 1.0
) -> npt.NDArray:
    processed_image = filters.unsharp_mask(image, radius=radius, amount=amount)
    return (processed_image * 255).astype(np.uint8)


def apply_gaussian_filter(image: npt.NDArray, sigma: int = 1) -> npt.NDArray:
    processed_image = filters.gaussian(image, sigma=sigma)
    return (processed_image * 255).astype(np.uint8)


def apply_wiener_filter(image: npt.NDArray, balance: float = 0.1) -> npt.NDArray:
    """Apply Wiener filter for image deconvolution.

    Args:
        image (npt.NDArray): The input image array.
        psf (npt.NDArray): The point spread function.
        balance (float): The regularization parameter.
        clip (bool): Whether to clip the output to the range [0, 1].

    Returns:
        npt.NDArray: The deconvolved image, scaled and cast to uint8.
    """

    def create_gaussian_psf(size: int = 5, sigma: float = 1.0) -> npt.NDArray:
        """Create a Gaussian PSF.

        Args:
            size (int): The size of the PSF array. Should be odd.
            sigma (float): The standard deviation of the Gaussian kernel.

        Returns:
            np.ndarray: A 2D array representing the Gaussian PSF.
        """
        if size % 2 == 0:
            size += 1

        ax = np.arange(-size // 2 + 1.0, size // 2 + 1.0)
        xx, yy = np.meshgrid(ax, ax)

        # Gaussian formula
        kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))

        return kernel / np.sum(kernel)

    psf = create_gaussian_psf()
    return restoration.wiener(image, psf, balance, clip=False)  # type: ignore


def apply_median_filter(
    image: npt.NDArray, kernel_size: int = 3
) -> npt.NDArray[np.uint8]:
    """
    Apply a median filter to an image using scipy.

    Args:
    image (npt.NDArray): The input image.
    kernel_size (int): The size of the neighborhood expressed as a single integer or a tuple of two integers.

    Returns:
    npt.NDArray: The denoised image.
    """
    processed_image = median_filter(image, size=kernel_size)
    return processed_image


def apply_bilateral_filter(
    image: npt.NDArray,
    sigma_color: float = 0.05,
    sigma_spatial: float = 1.5,
    bins: int = 127,
) -> npt.NDArray[np.uint8]:
    processed_image = restoration.denoise_bilateral(
        image, sigma_color=sigma_color, sigma_spatial=sigma_spatial, bins=bins
    )
    return (processed_image * 255).astype(np.uint8)


def apply_non_local_means(image: npt.NDArray, h: float = 1.0) -> npt.NDArray:
    processed_image = restoration.denoise_nl_means(image, h=h)
    return (processed_image * 255).astype(np.uint8)


@numba.jit(nopython=True)
def __local_adaptive_median_filter_func(
    window: npt.NDArray[np.float64], multiplier: float
) -> np.float64:
    center = window[len(window) // 2]
    S = np.sum(window)
    N = len(window)
    mu = S / N
    sigma = np.sqrt(np.sum((window - mu) ** 2) / N)
    LB = mu - multiplier * sigma
    UB = mu + multiplier * sigma

    # Create a mask for valid pixels within the bounds
    valid_mask = (window >= LB) & (window <= UB)
    valid_values = window[valid_mask]

    if len(valid_values) == 0:
        return np.median(window)
    else:
        return np.median(valid_values) if (center < LB or center > UB) else center


def apply_local_adaptive_median_filter(
    image: npt.NDArray[np.float64], radius: int = 1, multiplier: float = 1.0
) -> npt.NDArray[np.float64]:
    """
    Apply a local adaptive median filter to an image.

    Args:
        image (npt.NDArray): 2D array representing the grayscale image.
        radius (int): Radius of the sliding window around each pixel.
        multiplier (float): Multiplier for determining the noise threshold.

    Returns:
        npt.NDArray: The filtered image.
    """
    size = 2 * radius + 1
    filtered_image = generic_filter(
        image, __local_adaptive_median_filter_func,
        size=(size, size),
        extra_arguments=(multiplier,)
    )
    return filtered_image


@numba.jit(nopython=True)
def __compute_weights(
    patch: npt.NDArray, N: int, initial_K: float, local_mean: np.float64, local_std: np.float64
) -> npt.NDArray[np.float64]:
    center_pixel = patch[N, N]
    T_t0 = np.abs(center_pixel - local_mean) / local_std if local_std != 0 else 0
    weights = np.zeros_like(patch)

    for i in range(patch.shape[0]):
        for j in range(patch.shape[1]):
            current_pixel = patch[i, j]
            numerator = np.abs(current_pixel - center_pixel)
            abs_diffs = np.abs(patch - center_pixel)
            abs_diffs[N, N] = 0  # Exclude the center pixel from the sum
            total_pixels = (2 * N + 1) ** 2 - 1
            denominator = np.sum(abs_diffs) / total_pixels
            Q_sh = numerator / denominator if denominator != 0 else 0
            K = initial_K * T_t0 * Q_sh
            d_sh = np.sqrt((i - N) ** 2 + (j - N) ** 2)
            weights[i, j] = np.exp(-K * d_sh)
    
    return weights


def apply_frost_filter(
    image: npt.NDArray, window_size: int = 3, initial_K: float = 1.0
) -> npt.NDArray:
    """
    Source: https://ieeexplore.ieee.org/document/8844702
    """
    def adaptive_kernel(patch: npt.NDArray) -> np.float64:
        patch = patch.reshape(window_size, window_size)
        N = window_size // 2
        local_mean = np.mean(patch)
        local_std = np.std(patch)
        weights = __compute_weights(patch, N, initial_K, local_mean, local_std)
        weights /= np.sum(weights)
        return np.sum(weights * patch)

    filtered_image = generic_filter(image, adaptive_kernel, size=(window_size, window_size))
    filtered_image_uint8 = np.clip(filtered_image, 0, 255).astype(np.uint8)
    return filtered_image_uint8


def apply_wavelet_denoise_filter(
    image: npt.NDArray[np.float64],
    wavelet: str = "db1",
    mode: str = "soft",
) -> npt.NDArray[np.float64]:
    """
    Apply wavelet denoising to an image.

    Args:
    image (npt.NDArray[np.float64]): The input image as a numpy array (2D for grayscale or 3D for RGB).
    wavelet (str): The type of wavelet to use ('db1', 'db2', etc.).
    mode (str): Thresholding mode ('soft' or 'hard') used in wavelet thresholding.

    Returns:
    npt.NDArray[np.float64]: The denoised image as a numpy array.
    """
    if image.dtype != np.float64:
        image = img_as_float(image)

    processed_image = restoration.denoise_wavelet(
        image, wavelet=wavelet, mode=mode
    )

    return (processed_image * 255).astype(np.uint8)
