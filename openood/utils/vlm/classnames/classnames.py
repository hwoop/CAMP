from typing import List


_DATASET_CLASS_REGISTRY = {}


def register_classnames(
    name: str,
    classnames: List[str],
):

    if name in _DATASET_CLASS_REGISTRY:
        raise ValueError(f"Class names for dataset '{name}' are already registered.")

    _DATASET_CLASS_REGISTRY[name] = classnames


def get_class_names(
    dataset: str,
) -> List[str]:

    dataset = dataset.lower()
    if dataset not in _DATASET_CLASS_REGISTRY:
        available = sorted(_DATASET_CLASS_REGISTRY.keys())
        raise NotImplementedError(
            f"Unknown dataset '{dataset}'. Available datasets: {available}"
        )

    return _DATASET_CLASS_REGISTRY[dataset]
