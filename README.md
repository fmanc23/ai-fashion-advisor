# AI Fashion Advisor

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-FFD21E?logoColor=black)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

> Image-based outfit recommendation system that detects a person, segments the
> clothing they are wearing and suggests a replacement garment to improve the
> overall outfit — a virtual fashion advisor powered by computer vision.

Academic project for the **Computer Vision and Cognitive Systems** course
(MSc, University of Modena and Reggio Emilia — UNIMORE, 2023/2024).

🇬🇧 [English](#english) · 🇮🇹 [Italiano](#italiano)

---

## English

### Overview
The goal of this project is an outfit-recommendation system that, starting from
a single input image, suggests how to improve the outfit. It combines clothing
**segmentation** and person **detection** to make personalised fashion advice
accessible to everyone — particularly useful for people with little time or
interest in fashion, simplifying outfit selection and saving time and energy.

### How it works
The pipeline (see [`main.py`](main.py)) runs end-to-end:

1. **Pre-processing** — denoise and resize the input image (OpenCV).
2. **Person detection** — locate and crop the person with an **NVIDIA SSD** detector.
3. **Clothing segmentation** — segment each garment with **SegFormer-B2** (HuggingFace Transformers).
4. **Garment labelling** — classify each segmented mask via **OpenCLIP** (fashion CLIP) using text prompts.
5. **Compatibility analysis** — compute pairwise **cosine similarity** between garments to find the least-compatible item to replace.
6. **Recommendation** — score candidate replacements with an **Outfit Transformer** (fill-in-the-blank) over the **Polyvore** dataset and return the best match.
7. **Output** — the recommended item is shown to the user via `matplotlib`.

### Tech stack
- **Language:** Python
- **Deep learning:** PyTorch, TorchVision, HuggingFace Transformers
- **Models:** NVIDIA SSD (detection), SegFormer-B2 (segmentation), OpenCLIP / fashion CLIP (classification), Outfit Transformer (compatibility scoring)
- **Image processing:** OpenCV, scikit-image, albumentations, Pillow
- **Scientific:** NumPy, SciPy
- **Experiment tracking:** Weights & Biases (wandb)
- **Dataset:** Polyvore outfits

### Pre-trained weights
Download the [pre-trained weights](https://github.com/aimagelab/open-fashion-clip/releases/download/open-fashion-clip/finetuned_clip.pt)
and place the file inside `./src/open_fashion_clip/`.

> Note: model weights (`*.pt`) and the dataset are **not** tracked in this
> repository and must be downloaded separately.

### Getting started
Create the environment with conda:

```bash
conda env create -f environment.yml
conda activate cvcs-fashion
```

Run the pipeline from the project root:

```bash
python ./main.py
```

The input image goes in `./static/`; update the path variables in
[`main.py`](main.py) (around lines 150–151):

```python
img_path = "./static/image_test_2"
img_ext = ".jpg"
```

---

## Italiano

### Introduzione
L'obiettivo di questo progetto è sviluppare un sistema di raccomandazione di
outfit basato su immagini di input, sfruttando tecniche di segmentazione e
rilevamento dei vestiti. Lo scopo è fornire consigli di moda personalizzati,
creando un consulente di moda virtuale accessibile a tutti gli utenti tramite
applicazioni per smartphone o PC. Questo sistema automatizzato è particolarmente
vantaggioso per le persone con tempo limitato o poco interesse per la moda,
semplificando il processo di selezione dell'outfit e risparmiando tempo ed
energia preziosi.

### Come funziona
La pipeline (vedi [`main.py`](main.py)) esegue, in sequenza:

1. **Pre-elaborazione** — denoise e ridimensionamento dell'immagine (OpenCV).
2. **Rilevamento persona** — individuazione e ritaglio della persona con un detector **NVIDIA SSD**.
3. **Segmentazione capi** — segmentazione di ogni capo con **SegFormer-B2** (HuggingFace Transformers).
4. **Etichettatura capi** — classificazione di ogni maschera tramite **OpenCLIP** (fashion CLIP) con prompt testuali.
5. **Analisi di compatibilità** — calcolo della **similarità coseno** tra i capi per individuare quello meno compatibile da sostituire.
6. **Raccomandazione** — scoring dei capi candidati con un **Outfit Transformer** (fill-in-the-blank) sul dataset **Polyvore** e selezione del migliore.
7. **Output** — il capo consigliato viene mostrato all'utente tramite `matplotlib`.

### Pesi pre-addestrati
Scaricare i [pesi pre-addestrati](https://github.com/aimagelab/open-fashion-clip/releases/download/open-fashion-clip/finetuned_clip.pt)
e inserire il file all'interno della cartella `./src/open_fashion_clip/`.

> Nota: i pesi dei modelli (`*.pt`) e il dataset **non** sono inclusi nel
> repository e vanno scaricati a parte.

### Come iniziare
Creare l'ambiente con conda:

```bash
conda env create -f environment.yml
conda activate cvcs-fashion
```

Avviare la pipeline dalla cartella del progetto:

```bash
python ./main.py
```

L'immagine di input va inserita nella cartella `./static/`; nel file
[`main.py`](main.py) (righe 150–151 circa) vanno aggiornati i nomi delle
variabili:

```python
img_path = "./static/image_test_2"
img_ext = ".jpg"
```

---

## License
Distributed under the **Apache License 2.0**. See [`LICENSE`](LICENSE) for details.

## Authors
- [Nicola Guerra](https://github.com/Ng2k)
- [Francesco Mancinelli](https://github.com/fmanc23)
- [Davide Lupo](https://github.com/davidel01)
