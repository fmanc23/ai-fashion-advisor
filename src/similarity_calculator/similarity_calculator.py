"""
@author Nicola Guerra
@author Davide Lupo
@author Francesco Mancinelli
"""
import torch

from src.similarity_calculator.interface_similarity_function import ISimilarityFunction

class SimilarityFunction(ISimilarityFunction):
    """Classe per il calcolo della similarità
    """
    def __init__(self, function: ISimilarityFunction):
        self.function = function

    def compute_similarity(self, features) -> torch.Tensor:
        """Calcolo della similaritò tra vettori di features

        Args:
        -------
            features (list[np.ndarray]): array di features
        
        Returns:
        -------
            torch.Tensor: tensore con le similarità tra vettori di features
        """
        return self.function.compute_similarity(features)
