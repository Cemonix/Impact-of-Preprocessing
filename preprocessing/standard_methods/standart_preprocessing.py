import numpy as np
from numpy import typing as npt
from skimage import exposure, filters


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


def apply_clahe(image: npt.NDArray, clip_limit: float = 0.01, nbins: int = 256) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to the input image.

    Parameters:
        image (numpy.ndarray): The input image.
        clip_limit (float, optional): Threshold for contrast limiting. Default is 0.01.
        nbins (int, optional): Number of histogram bins. Default is 256.

    Returns:
        numpy.ndarray: The processed image after applying CLAHE.

    """
    processed_image = exposure.equalize_adapthist(image, clip_limit=clip_limit, nbins=nbins)
    return (processed_image * 255).astype(np.uint8)


def apply_otsu_thresholding(image: npt.NDArray, nbins: int = 256) -> npt.NDArray:
    """
    Applies Otsu thresholding to the input image.

    Parameters:
        image (numpy.ndarray): The input image.
        nbins (int): The number of bins used for histogram calculation. Default is 256.

    Returns:
        numpy.ndarray: The thresholded image.

    """
    threshold_value = filters.threshold_otsu(image, nbins=nbins)
    return (image > threshold_value).astype(np.uint8)


def apply_sobel_edge_detection(image: npt.NDArray) -> npt.NDArray:
    """
    Applies Sobel edge detection to the input image.

    Parameters:
        image (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The processed image with Sobel edge detection applied.
    """
    processed_image = filters.sobel(image)
    return (processed_image * 255).astype(np.uint8)

