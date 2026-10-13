from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping

from ..method_arguments import as_mapping
from .semantic import SEMANTIC_METHODS

COVARIANCE_MODES = ("full", "identity")
ID_REFERENCE_MODES = ("classwise", "global")
WEIGHTING_MODES = ("msp", "uniform")
DIAGNOSTIC_LEVELS = ("off", "samples", "summary", "full")


def _keys(
    node: Mapping[str, Any],
    allowed: tuple[str, ...],
    *,
    path: str,
) -> None:

    unknown = sorted(set(node) - set(allowed))
    if unknown:
        raise ValueError(f"{path}: unknown key(s) {unknown}; allowed keys are {list(allowed)}.")


def _int(
    node: Mapping[str, Any],
    name: str,
    default: int,
    *,
    path: str,
    minimum: int,
) -> int:

    value = node.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) != int(value):
        raise ValueError(f"{path}.{name} must be an integer, got {value!r}.")

    value = int(value)
    if value < minimum:
        raise ValueError(f"{path}.{name} must be >= {minimum}, got {value}.")

    return value


def _float(
    node: Mapping[str, Any],
    name: str,
    default: float,
    *,
    path: str,
) -> float:

    # str is accepted because YAML 1.1 reads an exponent without a dot (``1e-4``) as a string
    value = node.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{path}.{name} must be a number, got {value!r}.")

    return float(value)


def _choice(
    node: Mapping[str, Any],
    name: str,
    default: str,
    choices: tuple[str, ...],
    *,
    path: str,
) -> str:

    value = str(node.get(name, default))
    if value not in choices:
        raise ValueError(f"{path}.{name} must be one of {list(choices)}, got {value!r}.")

    return value


@dataclass(frozen=True)
class SemanticConfig:
    method: str = "mcm"
    args: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(
        cls,
        node: Mapping[str, Any],
    ) -> "SemanticConfig":

        _keys(node=node, allowed=("method", "args"), path="semantic")

        method = str(node.get("method", cls.method))
        if method not in SEMANTIC_METHODS:
            raise ValueError(f"semantic.method must be one of {list(SEMANTIC_METHODS)}, got {method!r}.")

        return cls(method=method, args=as_mapping(node.get("args"), path="semantic.args"))


@dataclass(frozen=True)
class CAMPConfig:
    """Settings of the ``camp`` block of ``postprocessor_args``."""

    window_size: int = 8192
    shift_eps: float = 1e-3
    min_samples: int = 512
    covariance: str = "full"
    id_reference: str = "classwise"
    weighting: str = "msp"

    @classmethod
    def parse(
        cls,
        node: Mapping[str, Any],
    ) -> "CAMPConfig":

        _keys(node=node, allowed=tuple(item.name for item in fields(cls)), path="camp")

        shift_eps = _float(node=node, name="shift_eps", default=cls.shift_eps, path="camp")
        if shift_eps <= 0.0:
            raise ValueError(f"camp.shift_eps must be > 0, got {shift_eps}.")

        return cls(
            window_size=_int(node=node, name="window_size", default=cls.window_size, path="camp", minimum=1),
            shift_eps=shift_eps,
            min_samples=_int(node=node, name="min_samples", default=cls.min_samples, path="camp", minimum=2),
            covariance=_choice(node=node, name="covariance", default=cls.covariance, choices=COVARIANCE_MODES, path="camp"),
            id_reference=_choice(node=node, name="id_reference", default=cls.id_reference, choices=ID_REFERENCE_MODES, path="camp"),
            weighting=_choice(node=node, name="weighting", default=cls.weighting, choices=WEIGHTING_MODES, path="camp"),
        )


@dataclass(frozen=True)
class FusionConfig:
    semantic_weight: float = 0.0

    @classmethod
    def parse(
        cls,
        node: Mapping[str, Any],
    ) -> "FusionConfig":

        _keys(node=node, allowed=("semantic_weight",), path="fusion")
        return cls(semantic_weight=_float(node=node, name="semantic_weight", default=cls.semantic_weight, path="fusion"))


@dataclass(frozen=True)
class DiagnosticsConfig:
    level: str = "off"
    output_dir_name: str = "camp_diagnostics"
    vector_checkpoints: tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256)
    spectrum_every: int = 10
    window_sample_rows: int = 0

    @classmethod
    def parse(
        cls,
        node: Mapping[str, Any],
    ) -> "DiagnosticsConfig":

        _keys(
            node=node,
            allowed=("level", "output_dir_name", "vector_checkpoints", "spectrum_every", "window_sample_rows"),
            path="diagnostics"
        )

        level = node.get("level", cls.level)
        if level is False:
            level = "off"

        level = str(level)
        if level not in DIAGNOSTIC_LEVELS:
            raise ValueError(f"diagnostics.level must be one of {list(DIAGNOSTIC_LEVELS)}, got {level!r} "
                "(quote the value in YAML: level: \"off\").")

        checkpoints = node.get("vector_checkpoints", cls.vector_checkpoints)
        if isinstance(checkpoints, (int, float)) and not isinstance(checkpoints, bool):
            checkpoints = (checkpoints,)
        elif isinstance(checkpoints, (list, tuple)):
            checkpoints = tuple(checkpoints)
        else:
            raise ValueError("diagnostics.vector_checkpoints must be an int or a list of ints.")

        checkpoints = tuple(
            _int(
                node={"vector_checkpoints": value},
                name="vector_checkpoints", default=0,
                path="diagnostics",
                minimum=0
            ) for value in checkpoints
        )

        return cls(
            level=level,
            output_dir_name=str(node.get("output_dir_name", cls.output_dir_name)),
            vector_checkpoints=tuple(sorted(set(checkpoints))),
            spectrum_every=_int(node=node, name="spectrum_every", default=cls.spectrum_every, path="diagnostics", minimum=1),
            window_sample_rows=_int(node=node, name="window_sample_rows", default=cls.window_sample_rows, path="diagnostics", minimum=0),
        )


    @property
    def enabled(self) -> bool:
        return self.level != "off"


    @property
    def records_batches(self) -> bool:
        return self.level in ("summary", "full")


    @property
    def records_vectors(self) -> bool:
        return self.level == "full"


@dataclass(frozen=True)
class CAMPPolicy:
    semantic: SemanticConfig
    camp: CAMPConfig
    fusion: FusionConfig
    diagnostics: DiagnosticsConfig

    @classmethod
    def from_config(
        cls,
        args: Any,
    ) -> "CAMPPolicy":

        if args is None:
            raise ValueError("postprocessor.postprocessor_args is required for the camp postprocessor.")

        node = as_mapping(args, path="postprocessor_args")
        _keys(node=node, allowed=("semantic", "camp", "fusion", "diagnostics"), path="postprocessor_args")

        def block(
            name: str,
        ) -> dict[str, Any]:
            return as_mapping(node.get(name), path=f"postprocessor_args.{name}")

        return cls(
            semantic=SemanticConfig.parse(block("semantic")),
            camp=CAMPConfig.parse(block("camp")),
            fusion=FusionConfig.parse(block("fusion")),
            diagnostics=DiagnosticsConfig.parse(block("diagnostics")),
        )


    def summary(self) -> dict[str, Any]:

        return {
            "semantic": {
                "method": self.semantic.method, "args": dict(self.semantic.args)
            },
            "camp": {
                "window_size": self.camp.window_size,
                "shift_eps": self.camp.shift_eps,
                "min_samples": self.camp.min_samples,
                "covariance": self.camp.covariance,
                "id_reference": self.camp.id_reference,
                "weighting": self.camp.weighting
            },
            "fusion": {
                "semantic_weight": self.fusion.semantic_weight
            },
            "diagnostics": {
                "level": self.diagnostics.level,
                "vector_checkpoints": list(self.diagnostics.vector_checkpoints),
                "spectrum_every": self.diagnostics.spectrum_every,
                "window_sample_rows": self.diagnostics.window_sample_rows
            }
        }


__all__ = [
    "COVARIANCE_MODES",
    "DIAGNOSTIC_LEVELS",
    "ID_REFERENCE_MODES",
    "WEIGHTING_MODES",
    "CAMPConfig",
    "CAMPPolicy",
    "DiagnosticsConfig",
    "FusionConfig",
    "SemanticConfig"
]
