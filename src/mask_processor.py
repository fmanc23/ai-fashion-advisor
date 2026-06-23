"""
Classe per la gestione delle maschere

@Author: Nicola Guerra
@Author: Davide Lupo
@Author: Francesco Mancinelli
"""
import numpy as np

class MaskProcessor:
    """
    Classe per processare le maschere di segmentazione.
    """
    @staticmethod
    def compute_masks(input_image: np.ndarray, segmented_image: np.ndarray) -> dict:
        """
        Computa le maschere per isolare i capi di abbigliamento dalla segmentazione.

        Args
        -------
            input_image : np.ndarray
                Immagine originale
            segmented_image : np.ndarray
                Immagine segmentata

        Returns
        -------
            ```python
            {
                label: np.ndarray
                ...
            }
            ```
                Le maschere per i capi di abbigliamento.
        """
        masks_dict = {}
        black_list = [0, 2, 11, 12, 13, 14, 15]
        right_shoe_index = 10
        left_shoe_index = 9
        last_mask = None
        for label in np.unique(segmented_image):
            if label not in black_list:
                mask = np.where(segmented_image == label, 1, 0)
                if label == right_shoe_index:
                    mask += last_mask
                if np.any(mask) and label != left_shoe_index:
                    masks_dict[label] = (input_image * mask[:, :, None]).astype(np.uint8)

                last_mask = mask

        return masks_dict
