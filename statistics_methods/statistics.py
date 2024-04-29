from typing import Tuple
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.image import extract_patches_2d


def estimate_noise_in_image(
    image: np.ndarray, patch_size: Tuple[int, int] = (7, 7), num_patches: int = 1000,
    lower_percentile: float = 0.1, upper_percentile: float = 0.2
) -> tuple:
    
    # Extract patches
    patches = extract_patches_2d(image, patch_size, max_patches=num_patches)

    # Reshape patches to a 2D array where each row is a patch
    patches = patches.reshape(patches.shape[0], -1).astype(np.float32)

    # Remove the mean
    patches -= np.mean(patches, axis=1, keepdims=True)

    # PCA to find the eigenvalues
    pca = PCA(n_components=patch_size[0]*patch_size[1])
    pca.fit(patches)

    # Calculate indices for the desired percentile range of eigenvalues
    lower_index = int(lower_percentile * len(pca.explained_variance_))
    upper_index = int(upper_percentile * len(pca.explained_variance_))

    # Noise variance estimate: median of eigenvalues within the specified percentile range
    noise_variance = np.median(pca.explained_variance_[lower_index:upper_index])

    return np.sqrt(pca.explained_variance_), np.sqrt(noise_variance)