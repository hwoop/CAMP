"""
Diagnostics recorder for CAMP runs, enabled through ``postprocessor_args.diagnostics``. 
It only observes a run and does not change any score.
"""
from __future__ import annotations

import atexit
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from openood.utils import get_config

from ..vlm_postprocessor import FrozenVLMBatch
from .config import CAMPPolicy
from .geometry import IDGeometry
from .scores import moments, projection_score
from .semantic import SemanticScore
from .window import StreamWindow, WindowRows


def run_metadata(
    postprocessor: Any,
) -> dict[str, Any]:

    config = postprocessor.config
    return {
        "method": "camp",
        "postprocessor": postprocessor.__class__.__name__,
        "semantic": dict(postprocessor.semantic.summary()),
        "resolved_config": postprocessor.policy.summary(),
        "seed": int(get_config(obj=config, name="seed", default=0) or 0),
        "dataset": postprocessor.dataset_name,
        "network": postprocessor.model_name,
        "backbone": postprocessor.backbone_name,
        "text_prompt": postprocessor.text_prompt,
        "text_center": bool(postprocessor.text_center),
        "class_num": int(postprocessor.class_num),
        "mark": str(get_config(obj=config, name="mark", default="")),
        "exp_name": str(get_config(obj=config, name="exp_name", default="")),
        "pipeline": str(get_config(obj=get_config(obj=config, name="pipeline", default=None), name="name", default="")),
        "evaluator": str(get_config(obj=get_config(obj=config, name="evaluator", default=None), name="name", default="")),
        "output_dir": str(get_config(obj=config, name="output_dir", default="")),
    }


def _np(
    value: torch.Tensor | None,
    dtype,
) -> np.ndarray | None:

    if value is None:
        return None

    return value.detach().cpu().numpy().astype(dtype)


def _git_commit(
    root: Path,
) -> str:

    try:
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False)

        return result.stdout.strip() if result.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


@dataclass
class _Stream:
    stream_id: str
    index: int
    metadata: dict[str, Any]


class CAMPDiagnosticRecorder:

    def __init__(
        self,
        policy: CAMPPolicy,
        *,
        output_dir: Path,
        repo_root: Path,
        run_metadata: Mapping[str, Any],
        semantic_method: str,
    ) -> None:

        self.policy = policy
        self.config = policy.diagnostics
        self.output_dir = Path(output_dir) / self.config.output_dir_name
        self.repo_root = Path(repo_root)
        self.run_metadata = dict(run_metadata)
        self.semantic_method = str(semantic_method)
        self._stream: _Stream | None = None
        self._stream_count = 0      # streams begun so far: the index of the next one
        self._batch_index = 0       # batches observed in the open stream
        self._sample: list[dict[str, Any]] = []
        self._batch: list[dict[str, Any]] = []
        self._vectors: dict[str, np.ndarray] = {}
        self._previous_direction: torch.Tensor | None = None
        self._previous_mu_id: torch.Tensor | None = None
        self._order = 0
        self._commit = ""
        if self.config.enabled:
            atexit.register(self.end_stream)


    def begin_stream(
        self,
        stream_id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:

        if not self.config.enabled:
            return

        self.end_stream()

        if not self._commit:
            self._commit = _git_commit(self.repo_root)

        index = self._stream_count
        self._stream = _Stream(
            stream_id=str(stream_id),
            index=index,
            metadata={
                "id_dataset": self.run_metadata.get("dataset", ""),
                "stream_index": index,
                **dict(metadata or {}),
            },
        )

        self._stream_count += 1
        self._batch_index = 0
        self._sample, self._batch, self._vectors = [], [], {}
        self._previous_direction = None
        self._previous_mu_id = None
        self._order = 0


    def end_stream(self) -> Path | None:

        if not self.config.enabled or self._stream is None:
            return None

        stream, self._stream = self._stream, None
        if not self._sample and not self._batch:
            return None  

        directory = self.output_dir / stream.stream_id
        directory.mkdir(parents=True, exist_ok=True)
        metadata = {
            **self.run_metadata,
            "commit": self._commit,
            "stream": {
                "stream_id": stream.stream_id,
                "stream_index": stream.index, **stream.metadata
            },
            "diagnostics": {
                "level": self.config.level,
                "observation_timing": "pre_update",
                "latency_scope": "host_scoring_before_window_update",
                "vector_checkpoints": list(self.config.vector_checkpoints),
                "spectrum_every": self.config.spectrum_every,
                "window_sample_rows": self.config.window_sample_rows
            },
            "rows": {
                "samples": sum(len(next(iter(block.values()))) for block in self._sample),
                "batches": len(self._batch)
            },
        }

        file = directory / "run_metadata.json"
        file.write_text(
            json.dumps(metadata, indent=2, sort_keys=True, default=str),
            encoding="utf-8"
        )

        self._write_blocks(self._sample, directory / "sample_scores")
        self._write_rows(self._batch, directory / "batch_diagnostics")

        if self._vectors:
            np.savez(directory / "selected_vectors.npz", **self._vectors)

        self._sample, self._batch, self._vectors = [], [], {}

        return directory


    @staticmethod
    def _write_frame(
        frame: pd.DataFrame,
        stem: Path,
    ) -> None:

        try:
            frame.to_parquet(stem.with_suffix(".parquet"), index=False)
        except (ImportError, ValueError, RuntimeError):
            frame.to_csv(stem.with_suffix(".csv"), index=False)


    @classmethod
    def _write_rows(
        cls,
        rows: list[dict[str, Any]],
        stem: Path,
    ) -> None:

        if rows:
            cls._write_frame(pd.DataFrame(rows), stem)


    @classmethod
    def _write_blocks(
        cls,
        blocks: list[dict[str, Any]],
        stem: Path,
    ) -> None:

        if not blocks:
            return

        names: list[str] = []

        for block in blocks:
            for name in block:
                if name not in names:
                    names.append(name)

        columns = {}

        for name in names:
            parts = []

            for block in blocks:
                value = block.get(name)
                if value is None:
                    reference = next(iter(block.values()))
                    value = np.full(shape=len(reference), fill_value=np.nan)

                parts.append(value)

            columns[name] = np.concatenate(parts)

        cls._write_frame(pd.DataFrame(columns), stem)


    @torch.no_grad()
    def observe(
        self,
        *,
        vlm: FrozenVLMBatch,
        semantic_score: SemanticScore,
        window: StreamWindow,
        current_windows: WindowRows | None,
        id_geometry: IDGeometry | None,
        camp_score: torch.Tensor | None,
        z_camp_score: torch.Tensor | None,
        z_semantic_score: torch.Tensor | None,
        final: torch.Tensor,
        diagnostic: Mapping[str, Any] | None,
        latency_ms: float,
    ) -> None:
        """Record one scored batch. Called before the batch is pushed into the window."""

        if not self.config.enabled:
            return

        if self._stream is None:
            self.begin_stream(f"stream_{self._stream_count:03d}")

        batch_index = self._batch_index
        self._batch_index += 1
        rows = int(vlm.features.shape[0])
        order = np.arange(self._order, self._order + rows, dtype=np.int64)
        self._order += rows
        stream_id = self._stream.stream_id

        undefined = np.full(shape=rows, fill_value=np.nan, dtype=np.float32)
        sample = {
            "stream_id": stream_id,
            "batch_idx": np.full(shape=rows, fill_value=batch_index, dtype=np.int32),
            "sample_idx": np.arange(rows, dtype=np.int32),
            "order_idx": order,
            "semantic_method": self.semantic_method,
            "S_native": _np(value=semantic_score.native, dtype=np.float32),
            "S_additive_axis": _np(value=semantic_score.logit, dtype=np.float32),
            "zS": _np(value=z_semantic_score, dtype=np.float32) if z_semantic_score is not None else undefined,
            "R_raw": _np(value=camp_score, dtype=np.float32) if camp_score is not None else undefined,
            "zR": _np(value=z_camp_score, dtype=np.float32) if z_camp_score is not None else undefined,
            "final": _np(value=final, dtype=np.float32),
            "defined": np.full(shape=rows, fill_value=z_camp_score is not None),
            "pred": _np(value=semantic_score.pred, dtype=np.int32),
        }

        if diagnostic is not None:                          # metadata supplied by the evaluator; not used for scoring
            label = diagnostic.get("label")
            is_ood = diagnostic.get("is_ood")

            if label is not None:
                sample["true_class"] = _np(value=label, dtype=np.int32)

            if is_ood is not None:
                sample["is_ood"] = _np(value=is_ood, dtype=np.int8)

            sample.update(
                _provenance("source_dataset", diagnostic.get("source_dataset"), rows)
            )

            sample.update(
                _provenance("source_index", diagnostic.get("source_index"), rows)
            )

        self._sample.append(_columns_to_rows(columns=sample, rows=rows))

        if not self.config.records_batches:
            return

        undefined_moments = (float("nan"), float("nan"))
        if z_semantic_score is not None:
            semantic_mean, semantic_var = moments(current_windows.semantic_logit)
        else:
            semantic_mean, semantic_var = undefined_moments
            
        if id_geometry is not None:
            camp_z_mean, camp_z_var = moments(projection_score(features=current_windows.features, u=id_geometry.u))
        else:
            camp_z_mean, camp_z_var = undefined_moments

        camp = self.policy.camp
        batch: dict[str, Any] = {
            "stream_id": stream_id,
            "batch_idx": batch_index,
            "order_idx": int(order[0]),
            "n_rows": rows,
            "defined": id_geometry is not None,
            "stream_count": int(window.seen),
            "effective_window_size": int(window.count),
            "window_size": int(window.window_size),
            "shift_eps": float(camp.shift_eps),
            "covariance_mode": camp.covariance,
            "id_reference_mode": camp.id_reference,
            "weighting_mode": camp.weighting,
            "semantic_mean": semantic_mean,
            "semantic_var": semantic_var,
            "camp_z_mean": camp_z_mean,
            "camp_z_var": camp_z_var,
            "state_bytes": int(window.state_bytes()),
            "latency_ms": float(latency_ms),
        }

        if id_geometry is not None:
            direction = id_geometry.u
            batch.update({
                "window_count": int(current_windows.count),
                "seen_class_count": int(id_geometry.k),
                "mu_stream_norm": float(id_geometry.mu_stream.norm()),
                "mu_id_norm": float(id_geometry.mu_id.norm()),
                "raw_shift_norm": float(id_geometry.delta.norm()),
                "precision_shift_norm": float(id_geometry.r.norm()),
                "cos_direction_prev": (float(direction @ self._previous_direction)
                    if self._previous_direction is not None else float("nan")),
                "mu_id_change": (float((id_geometry.mu_id - self._previous_mu_id).norm()
                        / id_geometry.mu_id.norm().clamp_min(1e-12))
                    if self._previous_mu_id is not None else float("nan")),
            })

            batch["n_classes_ge_5"] = self._classes_with_support(current_windows, 5)

            if batch_index % self.config.spectrum_every == 0:
                batch.update(self._spectrum(id_geometry.operator, id_geometry.eps))

            if self.config.records_vectors and batch_index in self.config.vector_checkpoints:
                self._checkpoint(stream_id, batch_index, id_geometry, current_windows)

            self._previous_direction = direction
            self._previous_mu_id = id_geometry.mu_id

        self._batch.append(batch)


    @staticmethod
    def _classes_with_support(
        current_windows: WindowRows,
        minimum: int,
    ) -> int:

        counts = torch.bincount(current_windows.pseudo_class.cpu())     # CUDA bincount is not deterministic
        return int((counts >= minimum).sum())


    @staticmethod
    def _spectrum(
        operator: torch.Tensor,
        eps: float,
    ) -> dict[str, float]:

        eigenvalues = torch.linalg.eigvalsh(operator)
        clamped = eigenvalues.clamp_min(0)
        return {
            "cov_eig_max": float(eigenvalues.max()),
            "cov_eig_min": float(eigenvalues.min()),
            "cov_condition": float(clamped.max() / clamped.min().clamp_min(1e-30)),
            "cov_effective_rank": float((clamped / (clamped + eps)).sum()),
            "cov_trace": float(torch.diagonal(operator).sum()),
        }


    def _checkpoint(
        self,
        stream_id: str,
        batch_index: int,
        id_geometry: IDGeometry,
        current_windows: WindowRows,
    ) -> None:

        prefix = f"{stream_id}/batch_{batch_index:06d}"
        self._vectors[f"{prefix}/mu_stream"] = _np(value=id_geometry.mu_stream, dtype=np.float32)
        self._vectors[f"{prefix}/mu_id"] = _np(value=id_geometry.mu_id, dtype=np.float32)
        self._vectors[f"{prefix}/delta"] = _np(value=id_geometry.delta, dtype=np.float32)
        self._vectors[f"{prefix}/direction_raw"] = _np(value=id_geometry.r, dtype=np.float32)
        self._vectors[f"{prefix}/direction_unit"] = _np(value=id_geometry.u, dtype=np.float32)
        self._vectors[f"{prefix}/covariance"] = _np(value=id_geometry.operator, dtype=np.float32)
        
        rows = self.config.window_sample_rows
        if rows > 0:
            take = min(rows, current_windows.count)
            step = max(current_windows.count // take, 1)
            index = torch.arange(0, take * step, step, device=current_windows.features.device)[:take]
            self._vectors[f"{prefix}/window_features"] = _np(value=current_windows.features[index], dtype=np.float32)
            self._vectors[f"{prefix}/window_weights"] = _np(value=current_windows.msp[index], dtype=np.float32)
            self._vectors[f"{prefix}/window_pseudo_class"] = _np(value=current_windows.pseudo_class[index], dtype=np.int32)


def _provenance(
    name: str,
    value: Any,
    rows: int,
) -> dict[str, Any]:

    if value is None:
        return {}

    if torch.is_tensor(value):
        if value.numel() != rows:
            return {}

        return {name: value.detach().cpu().numpy()}

    if isinstance(value, str):
        return {name: value}

    if isinstance(value, (list, tuple)):
        if len(value) != rows:
            return {}

        return {name: np.asarray(value)}

    return {}


def _columns_to_rows(
    columns: Mapping[str, Any],
    rows: int,
) -> dict[str, Any]:

    table = {
        key: value if isinstance(value, np.ndarray) else np.full(shape=rows, fill_value=value)
        for key, value in columns.items()
    }

    return table


__all__ = ["CAMPDiagnosticRecorder", "run_metadata"]
