"""
@Author Nicola Guerra
@Author: Davide Lupo
@Author: Francesco Mancinelli
"""
from os import path

import torch
import numpy as np
import open_clip
from PIL import Image

from src.features_extractor.interface_feature_extractor_model import IFeatureExtractorModel
from src.features_extractor.open_clip.options import OpenClipOptions
from src.utils.utils import get_device

class OpenClipModel(IFeatureExtractorModel):
    """Implementazione Open Clip
    """
    _MODEL_NAME = "ViT-B/32"
    _TOKENAZER_NAME = "ViT-B-32"
    _WEIGHTS_PATH = path.join(path.dirname(__file__), "finetuned_clip.pt")

    def __init__(self):
        self.device = get_device()
        self.model, self.preprocess, self.tokenizer = self.load_model({
            "model_name": self._MODEL_NAME,
            "weights_url": self._WEIGHTS_PATH,
            "tokenizer": self._TOKENAZER_NAME
        })

    def load_model(self, params: OpenClipOptions) -> tuple:
        model_name = params.get("model_name")
        weights_url = params.get("weights_url")
        tokenizer = params.get("tokenizer")

        model, _, preprocess = open_clip.create_model_and_transforms(model_name)
        state_dict = torch.load(weights_url, map_location = self.device)
        model.load_state_dict(state_dict["CLIP"])
        model = model.eval().requires_grad_(False).to(self.device)
        tokenizer = open_clip.get_tokenizer(tokenizer)
        return (model, preprocess, tokenizer)

    def load_and_process_image(self, image: np.ndarray) -> torch.Tensor:
        img = self.preprocess(Image.fromarray(image)).to(get_device())
        return img

    def _tokenize_prompts(self, prompts: list):
        """Tokenizza i prompt di testo

        Args:
        -------
            prompts (list): lista di prompt di testo

        Returns:
        -------
            _type_: _description_
        """
        tokenized_prompts = self.tokenizer(prompts).to(self.device)
        return tokenized_prompts

    def encode_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model.encode_image(image_tensor.unsqueeze(0))

    def encode_text(self, text_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model.encode_text(text_tensor)

    def _calculate_similarity(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return (100.0 * image_features @ text_features.T).softmax(dim=-1)

    def run_inference(self, image: torch.Tensor, prompts) -> torch.Tensor:
        text_tensor = self._tokenize_prompts(prompts)

        image_features = self.encode_image(image)
        text_features = self.encode_text(text_tensor)

        text_probs = self._calculate_similarity(image_features, text_features)
        return text_probs
