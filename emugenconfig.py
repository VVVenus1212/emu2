from transformers import PretrainedConfig
from typing import List


class EMUGENConfig(PretrainedConfig):

    def __init__(
        self,
        tokenizer = None,
        multimodal_encoder = None,
        scheduler = None,
        unet = None,
        vae = None,
        feature_extractor = None,
        safety_checker= None,
        **kwargs,
    ):

        self.tokenizer = tokenizer
        self.multimodal_encoder = multimodal_encoder
        self.scheduler = scheduler
        self.unet = unet
        self.vae = vae
        self.feature_extractor = feature_extractor
        self.safety_checker = safety_checker
        super().__init__(**kwargs)

