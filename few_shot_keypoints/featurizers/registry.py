# ---------------------------
# Featurizer registry utilities
# ---------------------------
from few_shot_keypoints.featurizers.base import BaseFeaturizer
class FeaturizerRegistry:
    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str | None = None):
        """
        Class decorator to register a BaseFeaturizer subclass under a string name.

        Usage:
            @FeaturizerRegistry.register("dift")
            class SDFeaturizer(BaseFeaturizer):
                ...

            # Or use NAME attribute on the class:
            @FeaturizerRegistry.register()
            class MyFeat(BaseFeaturizer):
                NAME = "my_feat"
        """
        def decorator(feat_cls: type):
            if not issubclass(feat_cls, BaseFeaturizer):
                raise TypeError("FeaturizerRegistry.register can only be used on BaseFeaturizer subclasses")
            key = (name or getattr(feat_cls, "NAME", None) or feat_cls.__name__).lower()
            if key in cls._registry and cls._registry[key] is not feat_cls:
                raise ValueError(f"Featurizer name already registered: {key}")
            cls._registry[key] = feat_cls
            return feat_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseFeaturizer:
        key = name.lower()
        if key not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(f"Unknown featurizer '{name}'. Available: {available}")
        return cls._registry[key](**kwargs)

    @classmethod
    def list(cls) -> list[str]:
        return sorted(cls._registry.keys())

