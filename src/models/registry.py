"""Generic Module Registry implementing Open/Closed and Dependency Inversion Principles."""

from typing import Any, Callable, Dict, List, Type, TypeVar

T = TypeVar("T")


class Registry:
    """
    Registry pattern allowing new architectural blocks or ablation baselines
    to be registered declaratively via decorators and instantiated from YAML configs
    without modifying orchestration code. Supports registering multiple alias names.
    """

    def __init__(self, registry_name: str) -> None:
        self._name = registry_name
        self._map: Dict[str, Type[Any]] = {}

    def register(self, *names: str) -> Callable[[Type[T]], Type[T]]:
        if not names:
            raise ValueError("At least one registration name must be provided.")

        def decorator(cls: Type[T]) -> Type[T]:
            for key in names:
                if key in self._map and self._map[key] is not cls:
                    raise KeyError(
                        f"Component '{key}' is already registered in {self._name}."
                    )
                self._map[key] = cls
            return cls

        return decorator

    def build(self, name: str, **kwargs: Any) -> Any:
        cls = self.get_class(name)
        return cls(**kwargs)

    def get_class(self, name: str) -> Type[Any]:
        if name not in self._map:
            available = ", ".join(sorted(self._map.keys()))
            raise KeyError(
                f"Unknown component '{name}' in registry '{self._name}'. "
                f"Available options: [{available}]"
            )
        return self._map[name]

    def keys(self) -> List[str]:
        return sorted(self._map.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._map


PRIOR_REGISTRY = Registry("PriorExtractor")
ENCODER_REGISTRY = Registry("MultiScaleEncoder")
ALIGNMENT_REGISTRY = Registry("AlignmentBlock")
FUSION_REGISTRY = Registry("TemporalFusionBlock")
DECONV_REGISTRY = Registry("DeconvBlock")
DECODER_REGISTRY = Registry("TwinDecoder")
