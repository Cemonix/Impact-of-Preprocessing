from pathlib import Path

from common.image_transforms import test_transformations

if __name__ == "__main__":
    image_path = Path("data/LungSegmentation/CXR_png/CHNCXR_0001_0.png")
    test_transformations(image_path)
