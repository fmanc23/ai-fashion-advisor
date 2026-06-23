# cvcs-fashion
Repository per il progetto dell'esame di Computer Vision and Cognitive Systems 2023/2024

## Table of Contents

- [cvcs-fashion](#cvcs-fashion)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Pesi pre-addestrati](#pesi-pre-addestrati)
  - [Get Started](#get-started)
  - [License](#license)
  - [Authors](#authors)

## Introduction

L'obiettivo di questo progetto è sviluppare un sistema di raccomandazione di outfit basato su immagini di input, sfruttando tecniche di segmentazione e rilevamento dei vestiti. Questo sforzo mira a fornire accesso a consigli di moda personalizzati, creando un consulente di moda virtuale accessibile a tutti gli utenti tramite applicazioni per smartphone o PC. Questo sistema automatizzato è particolarmente vantaggioso per le persone con tempo limitato o poco interesse per la moda, semplificando il processo di selezione dell'outfit e risparmiando tempo ed energia preziosi.

## Pesi pre-addestrati
Scaricare i [pesi pre-addestrati](https://github.com/aimagelab/open-fashion-clip/releases/download/open-fashion-clip/finetuned_clip.pt) e inserire il file all'interno della cartella `./src/open_fashion_clip/`

## Get Started
Installare le dipendenze esterne del progetto:

```
$ conda env create -f environment.yml
$ conda activate cvcs-fashion
```

Questo comando crea l'ambiente di sviluppo con anaconda

Per far partire la pipeline immettere il comando nella cartella del progetto

```
$ python ./main.py
```

L'output calcolato verrà mostrato all'utente mediante la libreria `matplotlib`

L'immagine di input è da inserire nella cartella `./static/` e nel file `main.py` alle righe 150 e 151 bisogna cambiare i nomi delle variabili.

```python
img_path = "./static/image_test_2"
img_ext = ".jpg"
```

## License

[Clicca qui per il file di license](./LICENSE.md)

## Authors

- [Nicola Guerra](https://github.com/Ng2k)
- [Francesco Mancinelli](https://github.com/fmanc23)
- [Davide Lupo](https://github.com/davidel01)