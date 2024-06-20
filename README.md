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


Tento repozitář obsahuje kód a data k diplomové práci zabývající se vlivem předzpracování obrazu a augmentace dat na segmentaci rentgenových snímků. Repozitář je strukturován následujícím způsobem:

## Struktura repozitáře
- **common/** - obsahuje pomocné funkce pro manipulaci s daty a pro vizualizaci.
- **configs/** - konfigurace pro jednotlivé modely a metody.
- **preprocessing/** - kód pro modely a metody předzpracování obrazu.
- **results/** - výsledky experimentů.
- **statistics_methods/** - statistické metody použité v projektu.
- **unet/** - kód pro modely U-Net a jejich trénink.
- **main.py** - hlavní soubor pro spuštění experimentů a trénink modelů.
- **poetry.lock** a **pyproject.toml** - závislosti a nastavení projektu pomocí Poetry.

## Poetry
Tento projekt používá nástroj Poetry pro správu závislostí a virtuálního prostředí. Poetry usnadňuje instalaci všech potřebných knihoven a zajišťuje, že projekt bude fungovat konzistentně na různých systémech.

### Nastavení projektu
Pro nastavení projektu postupujte podle následujících kroků:

1. Nainstalujte Poetry podle pokynů na [oficiálních stránkách](https://python-poetry.org/docs/#installation).
2. Naklonujte tento repozitář:

```bash
git clone https://github.com/Cemonix/Impact-of-Preprocessing.git
cd Impact-of-Preprocessing
```

3. Nainstalujte závislosti pomocí Poetry:

```bash
poetry install
```

4. Aktivujte virtuální prostředí:

```bash
poetry shell
```

## Spuštění experimentů
Pro spuštění hlavního souboru a zahájení experimentů použijte následující příkaz:

```bash
python main.py
```

## Průvodce hlavním souborem (`main.py`)
Hlavní soubor (`main.py`) slouží jako centrální bod pro spuštění různých experimentů a trénink modelů. Níže je popsáno, jak může uživatel soubor používat a jaké funkce obsahuje:

### Struktura hlavního souboru

`main.py` obsahuje několik funkcí, které se zaměřují na různé aspekty předzpracování dat a trénink modelů:

1. **create_dataset_main()**:

   - Vytvoří dataset se zašuměnými snímky pro trénink a testování.

2. **apply_model_and_create_dataset()**:

   - Použije vytrénovaný model pro odšumění snímků a vytvoří dataset s odšuměnými snímky.

3. **apply_ensemble_and_create_dataset()**:

   - Aplikuje metodu ensemble averaging na snímky a vytvoří dataset s odfiltrovanými snímky.

4. **train_unet_model()**:

   - Trénuje model U-Net pro binární segmentaci plicních snímků.

5. **test_unet_model()**:

   - Testuje vytrénovaný model U-Net na zvolených snímcích.

6. **train_multiclass_unet_model()**:

   - Trénuje model U-Net pro vícetřídní segmentaci zubních snímků.

7. **train_preprocessing_model()**:

   - Trénuje modely pro redukci šumu (např. DnCNN, DAE).

8. **test_preprocessing_model()**:

   - Testuje vytrénovaný model pro redukci šumu na zvolených snímcích.

9. **test_noise_transforms()**:

   - Testuje různé transformace šumu na snímcích.

10. **test_standard_preprocessing_methods()**:

    - Testuje standardní metody předzpracování snímků.

11. **test_preprocessing_ensemble_method()**:

    - Testuje metodu ensemble averaging pro předzpracování snímků.

12. **measure_metrics_for_images()**:

    - Měří metriky (např. PSNR, SSIM) pro odšuměné snímky.

13. **measure_noise_std()**:

    - Odhaduje směrodatnou odchylku šumu ve snímcích.

14. **plot_mlflow_runs_metrics()**:
    - Zobrazuje metriky z experimentů uložených v MLflow.

### Jak používat `main.py `

1. **Nastavení projektu**:

   - Před spuštěním experimentů je nutné nastavit virtuální prostředí a nainstalovat všechny závislosti pomocí Poetry.

2. **Spuštění experimentů**:

   - Každá funkce v `main.py ` představuje konkrétní experiment nebo krok v předzpracování dat. Pro spuštění konkrétní funkce je třeba ji odkomentovat v části `if __name__ == "__main__": ` a spustit soubor.

3. **Příklad spuštění tréninku modelu U-Net**:

   - Pro trénink modelu U-Net odkomentujte `train_unet_model() ` a spusťte hlavní soubor:
     ```bash
     poetry run python main.py
     ```

4. **Použití MLflow**:
   - Pro sledování experimentů a jejich metrik použijte MLflow. Spusťte MLflow pomocí:
     ```bash
     mlflow ui
     ```
   - Po spuštění tohoto příkazu bude MLflow UI dostupné na `http://localhost:5000 `, kde můžete procházet a analyzovat všechny experimenty provedené v rámci tohoto projektu.

Tento průvodce poskytuje základní přehled o tom, jak používat hlavní soubor `main.py ` k provádění různých experimentů a tréninku modelů v rámci projektu. Podrobnosti o jednotlivých funkcích a jejich parametrech lze najít přímo v kódu souboru.
