"""
La classe utilizza un modello di segmentation vestiti pre-addestrato.

@Author: Nicola Guerra
@Author: Davide Lupo
@Author: Francesco Mancinelli
"""

from PIL import Image
import torch

from src.segmentation.segmentation_model import SegmentationModel

class ClothesSegmentation():
    """
    Classe usata per rappresentare un modello SegformerB2Clothes.

    Attributi
    ----------
        _ssd_model : SSDModel
            Modello SSD.
    """

    _CONFIDENCE: float = 0.40

    def __init__(self, model: SegmentationModel):
        """
        Inizializza un nuovo oggetto SingleShotDetector.
        """
        self._segmentation_model: SegmentationModel = model

    def apply_segmentation(self, image: Image) -> torch.Tensor:
        """Applica la segmentazione all'immagine.

        Args:
        -------
            image (Image): immagine da processare

        Returns:
        -------
            torch.Tensor: immagine segmentata
        """
        return self._segmentation_model.apply_segmentation(image)
