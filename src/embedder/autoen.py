#/usr/bin/env python3

from src.embedder.base import BaseEmbedder


class AutoenEmbedder(BaseEmbedder):
    def __init__(
        self,
        **kwargs
        ) -> None:
        super().__init__(**kwargs)
        self.name = 'AutoEnEmbedder'
        self.training_style = 'autoencoder'
        assert self.training_style in {'autoencoder', 'decoding'}, f'{self.training_style} not supported'
        self._root_training_style = 'autoencoder'