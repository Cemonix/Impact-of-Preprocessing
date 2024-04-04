import numpy as np
from numpy import typing as npt
from skimage import exposure, filters, restoration
from scipy.ndimage import generic_filter

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
        kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))

        return kernel / np.sum(kernel)

    psf = create_gaussian_psf()
    return restoration.wiener(image, psf, balance, clip=False) # type: ignore


def apply_bilateral_filter(
    image: npt.NDArray, sigma_color: float = 0.05, sigma_spatial: int = 15
) -> npt.NDArray:
    processed_image = restoration.denoise_bilateral(
        image, sigma_color=sigma_color, sigma_spatial=sigma_spatial
    )
    return (processed_image * 255).astype(np.uint8)


def apply_non_local_means(image: npt.NDArray, h: float = 1.0) -> npt.NDArray:
    processed_image = restoration.denoise_nl_means(image, h=h)
    return (processed_image * 255).astype(np.uint8)


def apply_local_adaptive_median_filter(image: npt.NDArray, radius: int = 1, multiplier: float = 1.0) -> npt.NDArray:
    """
    Apply a local adaptive median filter to an image.

    Args:
        image (npt.NDArray): 2D array representing the grayscale image.
        radius (int): Radius of the sliding window around each pixel.
        multiplier (float): Multiplier for determining the noise threshold.

    Returns:
        npt.NDArray: The filtered image.
    """
    # Helper function to process each window
    def filter_func(window: npt.NDArray) -> np.floating:
        center = window[len(window) // 2]
        S = np.sum(window)
        N = (2 * radius + 1) ** 2
        mu = S / N
        sigma = np.sqrt(np.sum((window - mu) ** 2) / N)
        LB = mu - multiplier * sigma
        UB = mu + multiplier * sigma
        
        # Create a mask for valid pixels within the bounds
        valid_mask = (window >= LB) & (window <= UB)
        valid_values = window[valid_mask]
        
        if len(valid_values) == 0:  # If no valid values, return the median of the whole window
            return np.median(window)
        else:
            return np.median(valid_values) if (center < LB or center > UB) else center

    filtered_image = generic_filter(image, filter_func, size=(2*radius+1, 2*radius+1))

    return filtered_image