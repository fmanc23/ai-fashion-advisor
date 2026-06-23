"""
@author Nicola Guerra
@author Davide Lupo
@author Francesco Mancinelli
"""
from os import path, listdir
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

class PolyvoreDataset(Dataset):
    """Dataset per immagini Polyvore."""

    def __init__(self, images_dir: str, transform = None):
        """
        Args:
        -------
            images_dir (string): Percorso della directory con tutte le immagini.
            transform (callable, optional): Opzionale trasformazione da applicare
                                             su un campione.
        """
        self.images_dir = images_dir
        self.transform = transform
        self.filenames = [f for f in listdir(images_dir) if f.endswith(('.png', '.jpg'))]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = path.join(self.images_dir, self.filenames[idx])
        image = Image.open(img_path)
        image.resize((300, 300))

        return {
            'img_array': np.array(image.resize((300, 300))),
            'img_path': img_path
        }
