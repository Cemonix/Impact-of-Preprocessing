# Vliv předzpracování obrazu a augmentace dat na segmentaci rentgenových snímků 

<ul>
    <li>Teoretická část</li>
    <ol>
        <li>přehled metod předzpracování obrazu (vylepšení obrazu, redukce šumu, prahování aj.)</li>
        <li>vylepšení kvality rentgenových snímků pomocí neuronových sítí</li>
        <li>obecné postupy augmentace dat</li>
    </ol>
    <li>Praktická část</li>
    <ol start=4>
        <li>popis datasetu</li>
        <li>návrh metod vhodných pro předzpracování rentgenových snímků</li>
        <li>návrh a implementace specializovaných modelů neuronových sítí</li>
        <li>aplikace implementovaných metod a měření vlivu předzpracování na efektivitu učení segmentačních modelů</li>
        <li>zhodnocení výsledků</li>
    </ol>
</ul>

Spuštění MLflow
```
mlflow ui
```

1. Mean Squared Error (MSE): Measures the average squared difference between the denoised and the original clean images. Lower MSE values indicate better denoising performance. However, MSE may not always correlate well with perceived visual quality.

2. Peak Signal-to-Noise Ratio (PSNR): Commonly used in image processing, PSNR is a measure of the peak error between the original and the reconstructed image. Higher PSNR values generally indicate better denoising quality.

3. Structural Similarity Index (SSIM): SSIM is a perception-based model that considers changes in structural information, luminance, and contrast. Unlike MSE, SSIM can be more aligned with human visual perception. A higher SSIM value (close to 1) indicates better reconstruction quality.

#### 1. Image Enhancement:

Contrast Enhancement: Methods like histogram equalization or CLAHE (Contrast Limited Adaptive Histogram Equalization)
can significantly enhance the contrast in X-ray images, making important features more distinguishable.

#### 2. Noise Reduction:

Filters: Techniques like Gaussian, median, or bilateral filtering can help reduce noise. Gaussian filters are good for
reducing Gaussian noise, while median filters are particularly effective for salt-and-pepper noise. Bilateral filtering
is useful for noise reduction without smearing the edges.

#### 3. Edge Sharpening:

Edge Detection and Enhancement: Techniques like the Sobel or Laplacian filters can be used to detect and enhance edges.
After edge detection, you can add these edges back to the original image to make the edges sharper.

#### 4. Thresholding:

Binary and Adaptive Thresholding: This involves converting an image to a binary form, which can be useful in isolating
regions of interest. Binary thresholding sets pixels to either black or white based on a threshold value. Adaptive
thresholding, where the threshold value is determined based on the pixel's neighborhood, can be more effective for
uneven lighting conditions.

## Zdroje
https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels - hlavní datová sada
https://www.sciencedirect.com/science/article/abs/pii/S0734189X8780186X - Adaptive histogram equalization
