from __future__ import annotations

from typing import Any, Mapping, Sequence

from openood.utils import get_config

PER_DATASET_KEY = "per_dataset"
DEFAULT_SETTING = "default"


def as_mapping(
    value: Any,
    *,
    path: str,
) -> dict[str, Any]:

    if value is None:
        return {}

    if not hasattr(value, "keys"):
        raise ValueError(f"{path} must be a mapping, got {type(value).__name__}.")

    table = {
        str(key): value[key]
        for key in value.keys()
    }

    return table


def dataset_name(
    config: Any,
) -> str:
    return str(get_config(obj=get_config(obj=config, name="dataset", default=None), name="name", default=""))


def resolve_method_arguments(
    method: str,
    args: Any,
    defaults: Mapping[str, Any],
    *,
    dataset: str,
    required: Sequence[str] = (),
) -> tuple[dict[str, Any], str]:

    path = f"{method}.postprocessor_args"
    allowed = sorted(defaults)
    given = as_mapping(args, path=path)
    per_dataset = {
        str(name): as_mapping(node, path=f"{path}.{PER_DATASET_KEY}.{name}")
        for name, node in as_mapping(given.pop(PER_DATASET_KEY, None), path=f"{path}.{PER_DATASET_KEY}").items()
    }
    for where, node in [(path, given), *((f"{path}.{PER_DATASET_KEY}.{name}", node) for name, node in per_dataset.items())]:
        unknown = sorted(set(node) - set(allowed))
        if unknown:
            raise ValueError(f"{where}: unknown key(s) {unknown}; allowed keys are {allowed}.")

    values = {
        **defaults,
        **given,
    }
    setting = DEFAULT_SETTING

    if dataset in per_dataset:
        values.update(per_dataset[dataset])
        setting = f"{PER_DATASET_KEY}.{dataset}"

    missing = sorted(name for name in required if values.get(name) is None)
    if missing:
        raise ValueError(f"{path}: no value for {missing} on the ID dataset {dataset!r}.")

    return values, setting


def integer_argument(
    method: str,
    values: Mapping[str, Any],
    name: str,
    *,
    minimum: int,
) -> int:

    value = values[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{method}.postprocessor_args.{name} must be an integer >= {minimum}, got {value!r}.")

    return int(value)


def number_argument(
    method: str,
    values: Mapping[str, Any],
    name: str,
) -> float:

    value = values[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{method}.postprocessor_args.{name} must be a number, got {value!r}.")

    return float(value)


def boolean_argument(
    method: str,
    values: Mapping[str, Any],
    name: str,
) -> bool:

    value = values[name]
    if not isinstance(value, bool):
        raise ValueError(f"{method}.postprocessor_args.{name} must be a boolean, got {value!r}.")

    return value


__all__ = ["DEFAULT_SETTING", "PER_DATASET_KEY", "as_mapping", "boolean_argument", "dataset_name", "integer_argument",
    "number_argument", "resolve_method_arguments"]
