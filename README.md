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

## Zdroje
https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels - hlavní datová sada