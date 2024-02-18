import numpy as np
import numpy.typing as npt
from PIL import Image


def load_image_as_grayscale(path_to_image: str) -> npt.NDArray:
    """
    Load an image from a specified file path and convert it to a grayscale image.

    Args:
        path_to_image (str): The file path to the image to be loaded.

    Returns:
        npt.NDArray: A 2D NumPy array representing the grayscale image.
    """
    return np.asarray(Image.open(path_to_image).convert("L"))


def histogram_equalization_grayscale(image: npt.NDArray) -> npt.NDArray:
    """
    Perform histogram equalization on a grayscale image.

    This function applies histogram equalization to enhance the contrast of a
    grayscale image. It computes the histogram of the image, calculates the
    cumulative distribution function (CDF), and uses it to transform the pixel
    values. The transformed image is returned as a new NumPy array with enhanced
    contrast.

    Args:
        image (npt.NDArray): A 2D NumPy array representing the grayscale image
                             to be processed. It should be an 8-bit image with
                             values ranging from 0 to 255.

    Returns:
        npt.NDArray: A 2D NumPy array representing the histogram-equalized image.
    """
    histogram, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
    cdf = histogram.cumsum()
    cdf_min = cdf.min()

    widht, height = image.shape

    transformed_img: npt.NDArray = np.round(
        (cdf[image] - cdf_min) / (widht * height - cdf_min) * 255
    )
    return transformed_img.astype("uint8")


if __name__ == "__main__":
    img = load_image_as_grayscale("ChestImages/JPCNN001.jpg")
    equalized_image = histogram_equalization_grayscale(img)

    transformed_image = Image.fromarray(equalized_image)
    transformed_image.show()
