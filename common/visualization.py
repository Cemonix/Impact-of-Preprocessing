from typing import List
from PIL import Image
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sympy import Polygon


def compare_images(
    images: List[Image.Image],
    titles: List[str],
    images_per_column: int,
) -> None:
    assert len(titles) >= images_per_column, "Not enough titles!"

    # Calculate the number of rows needed for the given images per column
    rows = len(images) // images_per_column + (len(images) % images_per_column > 0)

    if len(titles) == images_per_column:
        subplot_titles = []
        for r in range(rows):
            for title in titles:
                subplot_titles.append(title if r == 0 else "")

    fig = make_subplots(rows=rows, cols=images_per_column, subplot_titles=titles)

    current_row, current_col = 1, 1
    for img in images:
        fig.add_trace(
            px.imshow(img, binary_string=True, color_continuous_scale="gray").data[0],
            row=current_row, col=current_col
        )
        current_col += 1
        if current_col > images_per_column:
            current_col = 1
            current_row += 1

    fig.update_layout(height=700, width=1800, title_text="Image Comparison")
    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    fig.show()


def display_polygon_on_image(
    images: Image.Image | List[Image.Image],
    polygons: List[Polygon]
) -> None:
    if isinstance(images, Image.Image):
        images = [images]

    fig = make_subplots(rows=len(images), cols=2, subplot_titles=["Image", "Image with prediction"])
    for idx, img in enumerate(images):
        fig.add_trace(
            px.imshow(img, binary_string=True, color_continuous_scale="gray").data[0],
            row=idx+1, col=1
        )
        fig.add_trace(
            px.imshow(img, binary_string=True, color_continuous_scale="gray").data[0],
            row=idx+1, col=2
        )
        for polygon in polygons:
            x, y = [], []
            for point in polygon.vertices:
                x.append(float(point.x))
                y.append(float(point.y))

            fig.add_trace(
                go.Scatter(x=x, y=y, mode='lines', fill='toself'),
                row=idx+1, col=2
            )

    fig.update_layout(height=800, width=1600, showlegend=False)
    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    fig.show()
    