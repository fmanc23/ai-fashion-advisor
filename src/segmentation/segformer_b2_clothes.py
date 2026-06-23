"""
Implementazione dell'interfaccia SegmentationModel per il modello SegformerB2Clothes.

Fornisce metodi per caricare il modello SegformerB2Clothes e le relative utility di elaborazione.

@Author Nicola Guerra
@Author: Davide Lupo
@Author: Francesco Mancinelli
"""
import torch
from torch import nn
from PIL import Image
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation

from src.segmentation.segmentation_model import SegmentationModel

class SegformerB2Clothes(SegmentationModel):
    """
    Classe concreta per il modello di segmentazione SegFormer_B2_Clothes.
    """

    _CONFIDENCE: float = 0.40

    def __init__(self):
        """
        Inizializza un nuovo oggetto SegformerB2Clothes.
        """
        model_name = "mattmdjaga/segformer_b2_clothes"
        self._processor = SegformerImageProcessor.from_pretrained(model_name)
        self._model = AutoModelForSemanticSegmentation.from_pretrained(model_name)

    def apply_segmentation(self, image: Image) -> torch.Tensor:
        inputs = self._processor(images=image, return_tensors="pt")

        outputs = self._model(**inputs)
        logits = outputs.logits.cpu()

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )

        pred_seg = upsampled_logits.argmax(dim=1)[0]

        return pred_seg
