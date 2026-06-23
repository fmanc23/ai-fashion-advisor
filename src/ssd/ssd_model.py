"""
Questo modulo definisce una classe base astratta per i modelli SSD (Single Shot MultiBox Detector).

Interfaccia che specifica i metodi che devono essere implementati da qualsiasi modello SSD concreto.

@Author Nicola Guerra
@Author: Davide Lupo
@Author: Francesco Mancinelli
"""
from abc import ABC, abstractmethod
import torch

class SSDModel(ABC):
    """
    Classe base astratta per i modelli SSD.

    Metodi
    -------
        load_image(self, image_url: str) -> dict
            Metodo astratto
            Carica l'immagine da un URL
        find_best_bboxes(self, image_tensor: torch.Tensor) -> list
            Metodo astratto
            Trova le migliori bounding box per l'immagine.
    """

    @abstractmethod
    def load_image(self, image_url: str) -> dict:
        """Carica l'immagine in 2 formati: numpy e tensore PyTorch.

        Args:
        -------
            image_url: str
              percorso immagine

        Returns:
        -------
            ```python
            {
                "image_numpy": np.ndarray,
                "image_tensor": torch.Tensor
            }
            ```:
              dizionario con l'immagine caricata come array numpy e tensore PyTorch
        """

    @abstractmethod
    def find_best_bboxes(self, image_tensor: torch.Tensor) -> list:
        """ Trova le migliori bounding box per l'immagine.

        Args:
        -------
            image (torch.Tensor): immagine da processare

        Returns:
        -------
            detection (list): lista delle detection
        """
