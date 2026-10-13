from __future__ import annotations

from importlib import import_module as _import_module
from typing import Final


_EXPORTS: Final[dict[str, tuple[str, str]]] = {
    "BasePostprocessor": (
        ".base_postprocessor",
        "BasePostprocessor"
    ),

    "get_postprocessor": (
        ".utils",
        "get_postprocessor"
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(
    name: str,
):

    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from exc

    module = _import_module(module_name, package=__name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
