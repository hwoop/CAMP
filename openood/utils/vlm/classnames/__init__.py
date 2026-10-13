from .classnames import (
    register_classnames,
    get_class_names,
)

from . import imagenet
from . import cifar

__all__ = [
    "register_classnames",
    "get_class_names",
]
