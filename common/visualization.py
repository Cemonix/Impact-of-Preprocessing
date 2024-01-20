import numpy as np
import plotly.express as px
from numpy import typing as npt


def compare_images(
    original: npt.NDArray,
    transformed: npt.NDArray,
    original_title: str,
    transformed_title: str,
) -> None:
    # Combine images into a sequence for displaying with imshow
    img_sequence = np.stack([original, transformed], axis=0)

    # Create the plot
    fig = px.imshow(
        img_sequence,
        facet_col=0,
        binary_string=True,
        color_continuous_scale="gray",
        labels={"facet_col": "Image"},
    )

    # Set titles for each facet
    fig.layout.annotations[0]["text"] = original_title
    fig.layout.annotations[1]["text"] = transformed_title

    fig.update_layout(height=800, width=1600, title_text="Image Comparison")
    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    fig.show()
