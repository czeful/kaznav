"""Name -> class registries for swappable components (VLM backends, detectors, encoders).

A component is selected in config either by a registered name (``type: qolda``)
or by an import path (``class_path: my_pkg.my_module:MyBackend``). The second
form lets a new model be added without touching this package at all.
"""
from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, Generic, List, Mapping, Type, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str, builtin_modules: List[str]):
        self.kind = kind
        self._builtin_modules = builtin_modules
        self._classes: Dict[str, Type[T]] = {}
        self._builtins_loaded = False

    def register(self, name: str) -> Callable[[Type[T]], Type[T]]:
        def deco(cls: Type[T]) -> Type[T]:
            if name in self._classes and self._classes[name] is not cls:
                raise ValueError(f"{self.kind} {name!r} is already registered to {self._classes[name]!r}")
            self._classes[name] = cls
            return cls

        return deco

    def _load_builtins(self) -> None:
        if not self._builtins_loaded:
            for mod in self._builtin_modules:
                importlib.import_module(mod)
            self._builtins_loaded = True

    def names(self) -> List[str]:
        self._load_builtins()
        return sorted(self._classes)

    def get(self, name: str) -> Type[T]:
        self._load_builtins()
        try:
            return self._classes[name]
        except KeyError:
            raise KeyError(f"unknown {self.kind} {name!r}; registered: {self.names()}") from None

    def resolve(self, spec: Mapping[str, Any]) -> Type[T]:
        """Class for a config block with either ``class_path`` or ``type``."""
        if spec.get("class_path"):
            module, _, attr = str(spec["class_path"]).partition(":")
            if not attr:
                raise ValueError(f"class_path must look like 'package.module:ClassName', got {spec['class_path']!r}")
            return getattr(importlib.import_module(module), attr)
        if "type" not in spec:
            raise KeyError(f"{self.kind} config needs 'type' or 'class_path'")
        return self.get(str(spec["type"]))
