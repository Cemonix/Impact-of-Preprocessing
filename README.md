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

Spuštění tensorboard
```
tensorboard --logdir=./lightning_logs/
```

1. Mean Squared Error (MSE): Measures the average squared difference between the denoised and the original clean images. Lower MSE values indicate better denoising performance. However, MSE may not always correlate well with perceived visual quality.

2. Peak Signal-to-Noise Ratio (PSNR): Commonly used in image processing, PSNR is a measure of the peak error between the original and the reconstructed image. Higher PSNR values generally indicate better denoising quality.

3. Structural Similarity Index (SSIM): SSIM is a perception-based model that considers changes in structural information, luminance, and contrast. Unlike MSE, SSIM can be more aligned with human visual perception. A higher SSIM value (close to 1) indicates better reconstruction quality.

## Zdroje
https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels - hlavní datová sada