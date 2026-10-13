from __future__ import annotations

from importlib import import_module as _import_module

_EXPORTS = {
    "CAMPPostprocessor": ".postprocessor",
    "StreamWindow": ".window",
    "IDGeometry": ".geometry",
    "id_geometry": ".geometry",
}

__all__ = list(_EXPORTS)


def __getattr__(
    name: str,
):

    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(_import_module(_EXPORTS[name], package=__name__), name)
    globals()[name] = value
    return value
