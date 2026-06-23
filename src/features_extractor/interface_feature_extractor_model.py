"""
@author Nicola Guerra
@author Davide Lupo
@author Francesco Mancinelli
"""

from abc import ABC, abstractmethod
import numpy as np
import torch

from src.features_extractor.model_options import IModelOptions

class IFeatureExtractorModel(ABC):
    """Interfaccia per il modello di estrazione feature e classificiazione maschere
    """

    @abstractmethod
    def load_model(self, params: IModelOptions) -> tuple:
        """
        Ritorna il modello, la funzione di pre-processing dell'immagine
        e il tokenizer per il prompt di testo

        Args:
        -------
            params(IModelOptions): parametri per il caricamento del modello

        Returns:
        -------
            tuple: modello, funzione di pre-processing, tokenizer
        """

    @abstractmethod
    def load_and_process_image(self, image: np.ndarray) -> torch.Tensor:
        """Carica e processa un'immagine

        Args:
        -------
            image (np.ndarray): vettore immagine

        Returns:
        -------
            torch.Tensor: immagine processata
        """

    @abstractmethod
    def encode_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Funzione di encode dell'immagine

        Args:
        ----------
            image_tensor (torch.Tensor): rappresentazione numpy dell'immagine

        Returns:
        ----------
            torch.Tensor | any: tensor dell'immagine codificato
        """

    @abstractmethod
    def encode_text(self, text_tensor: torch.Tensor) -> torch.Tensor:
        """Funzione di encode del prompt di testo

        Args:
        ----------
            text_tensor (torch.Tensor): tensor del prompt di testo

        Returns:
        ----------
            torch.Tensor | any: tensor del prompt di testo codificato
        """

    @abstractmethod
    def _calculate_similarity(self,
        image_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """Calcola il valore di similarità tra due vettori di features

        Args:
        -------
            features1 (list[np.ndarray]): vettore di tutte le features delle immagini
        
        Returns:
        -------
            torch.Tensor: tensore con le similarità tra vettori di features
        """

    @abstractmethod
    def run_inference(self, image: torch.Tensor, prompts) -> torch.Tensor:
        """Esegue l'inferenza

        Args:
        --------
            image (torch.Tensor): percorso dell'immagine
            prompts (list[str]): lista di prompt di testo

        Returns:
        --------
            torch.Tensor: tensore delle inference dell'immagine
        """
