from typing import List, Tuple
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def compare_images(
    images: List[Image.Image],
    titles: List[str],
    images_per_column: int = 3,
    rect_coords: Tuple[int, int, int, int] | None = None,
    zoom: bool = True,
) -> None:
    assert len(images) >= len(titles), "More titles than images!"
    assert (
        len(titles) >= images_per_column
    ), "Not enough titles for the given images per column!"

    image_x, image_y = images[0].size
    if rect_coords is None:
        rect_coords = (
            int(image_x * 0.25),
            int(image_y * 0.25),
            int(image_x * 0.05),
            int(image_y * 0.05),
        )

    # Calculate the number of rows needed for the given images per column
    rows = len(images) // images_per_column + (len(images) % images_per_column > 0)

    fig, axes = plt.subplots(rows, images_per_column, figsize=(15 * rows, 7 * rows))
    axes = axes.flatten()

    for idx, img in enumerate(images):
        ax = axes[idx]
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        if idx < len(titles):
            ax.set_title(titles[idx])

        if zoom:
            rect = patches.Rectangle(
                (rect_coords[0], rect_coords[1]),
                rect_coords[2],
                rect_coords[3],
                linewidth=2,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)

            cropped_image = img.crop(
                (
                    rect_coords[0],
                    rect_coords[1],
                    rect_coords[0] + rect_coords[2],
                    rect_coords[1] + rect_coords[3],
                )
            )

            ax_inset = ax.inset_axes([0.7, 0.0, 0.3, 0.3])
            ax_inset.imshow(cropped_image, cmap="gray")
            ax_inset.axis("off")

    for ax in axes[len(images) :]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
