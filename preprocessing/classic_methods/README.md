#### 1. Image Enhancement:  
Contrast Enhancement: Methods like histogram equalization or CLAHE (Contrast Limited Adaptive Histogram Equalization) can significantly enhance the contrast in X-ray images, making important features more distinguishable.

#### 2. Noise Reduction:  
Filters: Techniques like Gaussian, median, or bilateral filtering can help reduce noise. Gaussian filters are good for reducing Gaussian noise, while median filters are particularly effective for salt-and-pepper noise. Bilateral filtering is useful for noise reduction without smearing the edges.

#### 3. Edge Sharpening:  
Edge Detection and Enhancement: Techniques like the Sobel or Laplacian filters can be used to detect and enhance edges. After edge detection, you can add these edges back to the original image to make the edges sharper.

#### 4. Thresholding:  
Binary and Adaptive Thresholding: This involves converting an image to a binary form, which can be useful in isolating regions of interest. Binary thresholding sets pixels to either black or white based on a threshold value. Adaptive thresholding, where the threshold value is determined based on the pixel's neighborhood, can be more effective for uneven lighting conditions.

## Sources:
https://www.sciencedirect.com/science/article/abs/pii/S0734189X8780186X - Adaptive histogram equalization