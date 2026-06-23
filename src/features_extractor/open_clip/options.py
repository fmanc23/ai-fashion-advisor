"""
@author Nicola Guerra
@author Davide Lupo
@author Francesco Mancinelli
"""
from dataclasses import dataclass

from src.features_extractor.model_options import IModelOptions

@dataclass
class OpenClipOptions(IModelOptions):
    """Classe per definizione opzioni del modello OpenCLIP

    ```python
    {
        "model_name": str,
        "weights_url": str,
        "tokenizer": str
    }
    ```

    Attributes:
    ------
        model_name (str): nome del modello
        weights_url (str): url dei pesi del modello
        tokenizer (str): nome del tokenizer
    """
    tokenizer: str
