"""
Read recorded run configurations and YAML files, 
and apply dotted overrides to plain config trees.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, MutableMapping

import yaml

if TYPE_CHECKING:
    from openood.utils.config import Config

PYTHON_TAG_PREFIX = "tag:yaml.org,2002:python/"
_OBJECT_PAYLOAD_KEYS = ("dictitems", "state")


class RecordedConfigLoader(yaml.SafeLoader):
    """
    SafeLoader that reads every ``!!python/...`` tag as the plain mapping,
    sequence or scalar it wraps.
    """


def _construct_python_tag(
    loader: yaml.SafeLoader,
    suffix: str,
    node: yaml.Node,
) -> Any:

    del suffix

    if isinstance(node, yaml.MappingNode):
        mapping = loader.construct_mapping(node, deep=True)

        for key in _OBJECT_PAYLOAD_KEYS:
            if key in mapping and isinstance(mapping[key], Mapping):
                return dict(mapping[key])

        return mapping

    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)

    return loader.construct_scalar(node)


RecordedConfigLoader.add_multi_constructor(PYTHON_TAG_PREFIX, _construct_python_tag)


def load_yaml(
    path: Path | str,
    *,
    python_tags: bool = False,
) -> Any:

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.load(handle, Loader=RecordedConfigLoader if python_tags else yaml.SafeLoader)


def load_recorded_config(
    run_dir: Path | str,
) -> Config:

    from openood.utils.config import Config

    path = Path(run_dir)
    if path.is_dir():
        path = path / "config.yml"

    if not path.is_file():
        raise FileNotFoundError(f"no recorded configuration at {path}")

    tree = load_yaml(path, python_tags=True)
    if not isinstance(tree, Mapping):
        raise ValueError(f"{path} does not hold a configuration mapping (got {type(tree).__name__}).")

    return Config(dict(tree))


def merge_trees(
    base: Mapping[str, Any],
    overlay: Mapping[str, Any],
) -> dict[str, Any]:

    merged: dict[str, Any] = {key: value for key, value in base.items()}

    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = merge_trees(base=merged[key], overlay=value)
        else:
            merged[key] = value

    return merged


__all__ = [
    "PYTHON_TAG_PREFIX",
    "RecordedConfigLoader",
    "load_recorded_config",
    "load_yaml",
    "merge_trees",
]
