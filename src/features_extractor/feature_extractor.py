"""
@author Nicola Guerra
@author Davide Lupo
@author Francesco Mancinelli
"""
from src.features_extractor.interface_feature_extractor_model import IFeatureExtractorModel

class FeatureExtractor:
    """Classe per l'estrazione delle feature dalle maschere delle immagini
    """
    def __init__(self, model: IFeatureExtractorModel):
        self.model = model

    def extract_masks_features(self, masks: dict) -> list:
        """Estrae vettori delle feature dalle maschere delle immagini

        Args:
        -------
            masks (`{<key>: <np.ndarray> [...]}`): dizionario delle maschere
        """
        image_features = []
        for mask in masks.values():
            mask_tensor = self.model.load_and_process_image(mask)
            feature_vector = self.model.encode_image(mask_tensor)
            # Aggiungi una dimensione batch per ogni vettore
            image_features.append(feature_vector.unsqueeze(0))

        return image_features
