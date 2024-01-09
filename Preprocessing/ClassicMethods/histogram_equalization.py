import numpy as np
from PIL import Image

import numpy.typing as npt

def load_image_as_grayscale(path_to_image: str) -> npt.NDArray:
    """
    Load an image from a specified file path and convert it to a grayscale image.

    Args:
        path_to_image (str): The file path to the image to be loaded.

    Returns:
        npt.NDArray: A 2D NumPy array representing the grayscale image.
    """
    return np.asarray(Image.open(path_to_image).convert('L'))

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
    histogram, _ = np.histogram(image.flatten(), bins=256, range=[0, 256])
    cdf = histogram.cumsum()
    cdf_min = cdf.min()

    M, N = image.shape

    transformed_img: npt.NDArray = np.round(
        (cdf[image] - cdf_min) / (M * N - cdf_min) * (255)
    )
    return transformed_img.astype('uint8')

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def visualize_he_histograms(original_image: npt.NDArray, transformed_image: npt.NDArray) -> None:
    """
    Visualize the histograms of the original and transformed images using Plotly.

    Args:
        original_image (npt.NDArray): A 2D NumPy array representing the original grayscale image.
        transformed_image (npt.NDArray): A 2D NumPy array representing the transformed grayscale image.
    """

    # Calculate histograms
    original_histogram, _ = np.histogram(
        original_image.flatten(), bins=256, range=[0, 256]
    )
    transformed_histogram, _ = np.histogram(
        transformed_image.flatten(), bins=256, range=[0, 256]
    )

    cumulative_original_histogram = original_histogram.cumsum()
    cumulative_original_histogram_normalized = (
        cumulative_original_histogram / cumulative_original_histogram[-1]
    ) * original_histogram.max()

    cumulative_transformed_histogram = transformed_histogram.cumsum()
    cumulative_transformed_histogram_normalized = (
        cumulative_transformed_histogram / cumulative_transformed_histogram[-1]
    ) * transformed_histogram.max()

    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Original Image Histogram", "Transformed Image Histogram")
    )

    # Add histograms to subplots
    fig.add_trace(
        go.Bar(
            x=list(range(256)), y=original_histogram, marker_color='blue', name='Histogram'
        ), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(256)), y=cumulative_original_histogram_normalized, mode='lines',
            name='Cumulative Histogram', line=dict(color='green')
        ), row=1, col=1
    )

    fig.add_trace(
        go.Bar(
            x=list(range(256)), y=transformed_histogram, marker_color='red', name='Histogram'
        ), row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(256)), y=cumulative_transformed_histogram_normalized, mode='lines',
            name='Cumulative Histogram', line=dict(color='orange')
        ), row=1, col=2
    )

    # Update layout
    fig.update_layout(
        height=840, width=1660, title_text="Histogram Comparison", bargap=0.2
    )
    fig.show()

if __name__ == '__main__':
    image = load_image_as_grayscale('ChestImages/JPCNN001.jpg')
    equalized_image = histogram_equalization_grayscale(image)

    visualize_he_histograms(image, equalized_image)

    transformed_image = Image.fromarray(equalized_image)
    transformed_image.show()

    # TODO: CLAHE