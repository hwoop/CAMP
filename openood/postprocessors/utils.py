from __future__ import annotations

from importlib import import_module as _import_module
from typing import Any, Final, Mapping

from openood.utils import Config
from openood.utils.config import merge_configs


_POSTPROCESSOR_IMPORTS: Final[dict[str, tuple[str, str]]] = {
    "camp": (".camp.postprocessor", "CAMPPostprocessor"),
    "mcm": (".mcm.postprocessor", "MCMPostprocessor"),
    "neglabel": (".neglabel.postprocessor", "NegLabelPostprocessor"),
    "csp": (".csp.postprocessor", "CSPPostprocessor"),
    "tanl": (".tanl.postprocessor", "TANLPostprocessor"),
    "adaneg": (".adaneg.postprocessor", "AdaNegPostprocessor"),
    "oodd": (".oodd.postprocessor", "OODDPostprocessor"),
    "coevo": (".coevo.postprocessor", "CoEvoPostprocessor"),
}


def get_postprocessor(
    config: Config,
):

    name = str(config.postprocessor.name)

    try:
        module_name, class_name = _POSTPROCESSOR_IMPORTS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown postprocessor: {name!r}") from exc

    module = _import_module(module_name, package=__package__)
    postprocessor_type = getattr(module, class_name)
    return postprocessor_type(config)


def get_inner_postprocessor(
    config: Config,
    name: str,
    args: Mapping[str, Any],
):
    """
    Build the postprocessor of another method for the same run, 
    from the run's config with only the ``postprocessor`` block replaced.
    """

    inner = merge_configs(config)
    if "postprocessor" in inner:
        del inner["postprocessor"]

    inner["postprocessor"] = Config({"name": name, "postprocessor_args": dict(args)})
    return get_postprocessor(inner)


__all__ = ["get_inner_postprocessor", "get_postprocessor"]
