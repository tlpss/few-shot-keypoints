import torch
import abc

class BaseFeaturizer(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def extract_features(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        # image should be 1,C,H,W tensor.
        # output will be 1,D,H,W tensor.
        raise NotImplementedError("Subclasses must implement this method")