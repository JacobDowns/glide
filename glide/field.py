from dataclasses import dataclass, field, fields
from typing import Any
import cupy as cp

@dataclass
class Field:
    data: Any
    name: str | None = None
    units: str | None = None
    attrs: dict = field(default_factory=dict)
    _grad: Any | None = None
    initialized: bool = False

    def set(self, value) -> None:
        if hasattr(value, "shape"):
            self.data[...] = cp.array(value,dtype=cp.float32)
        else:
            self.data.fill(value)
        self.initialized = True

    def zero(self) -> None:
        self.data.fill(0)

    @property
    def grad(self):
        if self._grad is None:
            self._grad = self._zeros_like()
        return self._grad

    def has_grad(self) -> bool:
        return self._grad is not None

    def zero_grad(self) -> None:
        if self._grad is not None:
            self._grad.fill(0)

    def _zeros_like(self):
        return cp.zeros_like(self.data)

    def __repr__(self):
        string = f'Field: {self.name}\n{self.data}\n{self.data.shape}, {self.data.dtype}, {self.units}\nInitialized: {self.initialized}'
        return string

    @property
    def compact_string(self):
        string = f'Field: {self.name}, {self.units}, ({self.data.shape[0]}, {self.data.shape[1]})'
        return string

@dataclass
class Constant:
    value: Any
    name: str | None = None
    units: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    _grad: Any | None = None

    def set(self, value) -> None:
        self.value = cp.float32(value)

    @property
    def grad(self):
        if self._grad is None:
            self._grad = 0.0
        return self._grad

    def has_grad(self) -> bool:
        return self._grad is not None

    def zero_grad(self) -> None:
        if self._grad is not None:
            self._grad = 0.0

    def __repr__(self):
        string = f'Constant: {self.name}, {self.value:.3f}, {self.units}'
        return string

@dataclass
class SubgridField(Field):
    quantiles: Any | None = None

class LocalOption:
    def __init__(self, getter, setter, name: str):
        self._getter = getter
        self._setter = setter
        self._name = name

    def get(self):
        return self._getter()

    def set(self, value):
        self._setter(value)

    def __repr__(self):
        return f"{self._name}={self.get()!r}"


class BroadcastOption:
    def __init__(self, levels, getter, attr_name: str):
        self._levels = levels
        self._getter = getter
        self._attr_name = attr_name

    def get(self):
        # Representative value from the first level
        return getattr(self._getter(self._levels[0]), self._attr_name)

    def set(self, value):
        for lev in self._levels:
            cfg = self._getter(lev)
            setattr(cfg, self._attr_name, value)

    def __repr__(self):
        return f"{self._attr_name}={self.get()!r}"
