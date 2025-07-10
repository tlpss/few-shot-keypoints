from typing import Callable
import torch
import abc
import functools
import inspect

class FeaturizerCache:
    """
    A decorator to cache the last image and features for a BaseFeaturizer subclass. 
    Particulary useful if combined with KeypointsMatcher, because they would extract features for the same image multiple times.
    """
    def __init__(self, func: Callable):

        self.func = func
        # check that this is the extract_features method for a subclass of BaseFeaturizer
        method_name = func.__name__
        if not method_name == BaseFeaturizer.extract_features.__name__:
            raise ValueError("FeaturizerCache must be used as a decorator on the extract_features method of a subclass of BaseFeaturizer")
       
        self.last_image = None
        self.last_features = None


    def __get__(self, instance, owner) -> Callable:

        bound_func = functools.partial(self.func, instance) # bind the instance to the function (ie. add 'self' to the function)
        # can also use self.func.__get__(instance, owner)
        @functools.wraps(bound_func)
        def wrapper(*args, **kwargs):
            img = args[0]
            if self.last_image is not None and self.last_image.shape == img.shape and torch.allclose(self.last_image, img):
                # using cached features
                #print(f"using cached features")
                return self.last_features
            else:
                self.last_image = img.clone()
                self.last_features = bound_func(*args, **kwargs).clone()
                return self.last_features
        return wrapper


class BaseFeaturizer(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def extract_features(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        # image should be 1,C,H,W tensor.
        # output will be 1,D,H,W tensor.
        raise NotImplementedError("Subclasses must implement this method")

if __name__ == "__main__":
    import time
    class TestFeaturizer(BaseFeaturizer):
        @FeaturizerCache
        def extract_features(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
            return torch.zeros(1,1,24)

    featurizer = TestFeaturizer()
    img = torch.randn(1,3,512,512)
    time_start = time.time()
    print(featurizer.extract_features(img).shape)
    time_end = time.time()
    print(f"time taken: {time_end - time_start}")
    time_start = time.time()
    print(featurizer.extract_features(img).shape)
    time_end = time.time()
    print(f"time taken: {time_end - time_start}")