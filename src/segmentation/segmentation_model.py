"""
Questo modulo definisce una classe base astratta per i modelli di segmentazione.

Interfaccia che specifica i metodi che devono essere implementati da qualsiasi modello concreto.

@Author Nicola Guerra
@Author: Davide Lupo
@Author: Francesco Mancinelli
"""
from abc import ABC, abstractmethod
from PIL import Image
import torch

class SegmentationModel(ABC):
    """
    Classe base astratta per i modelli di segmentazione.

    Metodi
    -------
        apply_segmentation(self, image: Image) -> dict
            Metodo astratto
            Carica l'immagine da un URL
    """

    @abstractmethod
    def apply_segmentation(self, image: Image) -> torch.Tensor:
        """Applica la segmentazione all'immagine.

        Args:
        -------
            image (Image): immagine da processare

        Returns:
        -------
            torch.Tensor: immagine segmentata
        """
