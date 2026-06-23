"""
@author Nicola Guerra
@author Davide Lupo
@author Francesco Mancinelli
"""
from dataclasses import dataclass

@dataclass
class IModelOptions:
    """Classe per definizione opzioni di un modello di inferenza
    """
    model_name: str
    weights_url: str
