from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from openood.utils import get_config

from ..vlm_postprocessor import VLMPostprocessor, build_frozen_vlm_batch
from .config import CAMPPolicy
from .diagnostics import CAMPDiagnosticRecorder
from .window import StreamWindow
from .geometry import IDGeometry

from . import diagnostics
from . import geometry
from . import scores
from . import semantic


class CAMPPostprocessor(VLMPostprocessor):
    observes_provenance = True

    def __init__(
        self,
        config,
    ):

        super().__init__(config)

        self.recorder: CAMPDiagnosticRecorder | None = None
        self.geometry_args: dict[str, Any] | None = None
        self._id_prototypes: torch.Tensor | None = None

        self.policy = CAMPPolicy.from_config(self.args)

        self.semantic = semantic.SemanticPlugin(
            config=self.config,
            method=self.policy.semantic.method,
            args=self.policy.semantic.args,
        )

        self.window = StreamWindow(
            window_size=self.policy.camp.window_size,
            min_samples=self.policy.camp.min_samples,
        )


    def setup(
        self,
        net: nn.Module,
        id_loader_dict=None,
        ood_loader_dict=None,
    ):

        super().setup(net, id_loader_dict, ood_loader_dict)

        self.semantic.setup(net, id_loader_dict, ood_loader_dict)

        diag_config = self.policy.diagnostics
        camp_config = self.policy.camp

        self.geometry_args = dict(
            eps=camp_config.shift_eps,
            class_num=int(self.class_num),
            covariance=camp_config.covariance,
            id_reference=camp_config.id_reference,
            weighting=camp_config.weighting,
        )

        self._id_prototypes = F.normalize(self.id_txt_feat, dim=0).float()
        if diag_config.enabled and self.recorder is None:
            self.recorder = CAMPDiagnosticRecorder(
                self.policy,
                output_dir=Path(str(get_config(obj=self.config, name="output_dir", default="./results"))),
                repo_root=Path(__file__).resolve().parents[3],
                run_metadata=diagnostics.run_metadata(self),
                semantic_method=self.semantic.name,
            )

        print(
            f"[CAMPPostprocessor] setup complete | semantic={dict(self.semantic.summary())}, "
            f"window_size={camp_config.window_size}, shift_eps={camp_config.shift_eps}, min_samples={camp_config.min_samples}, "
            f"covariance={camp_config.covariance}, id_reference={camp_config.id_reference}, weighting={camp_config.weighting}, "
            f"semantic_weight={self.policy.fusion.semantic_weight}, diagnostics={diag_config.level}, "
            f"class_num={self.class_num}",
            flush=True,
        )


    def reset_memory(self) -> None:

        super().reset_memory()
        self.window.reset()
        self.semantic.reset()


    @torch.no_grad()
    def postprocess(
        self,
        net: nn.Module,
        data: Any,
        diagnostic: Optional[Mapping[str, Any]] = None,
    ):

        if self.geometry_args is None:
            raise RuntimeError("setup() must be called before postprocess().")

        started = time.perf_counter()

        vlm = build_frozen_vlm_batch(
            raw=self._extract_image_features(net, data),
            logit_scale=self.logit_scale,
            id_prototypes=self._id_prototypes,
        )

        current_windows = self.window.rows()

        semantic_score = self.semantic.score(
            net=net,
            image_features=vlm.raw,
        )

        z_semantic_score = None

        if current_windows is not None and current_windows.count >= 2:
            z_semantic_score = scores.standardize(
                values = semantic_score.logit,
                reference = current_windows.semantic_logit
            )

        camp_score = None
        z_camp_score = None
        id_geometry: IDGeometry = None

        if self.window.ready:
            id_geometry = geometry.id_geometry(
                features = current_windows.features,
                msp = current_windows.msp,
                pseudo_class = current_windows.pseudo_class,
                **self.geometry_args,
            )

            camp_score = scores.projection_score(vlm.features, id_geometry.u)
            z_camp_score = scores.standardize(
                values = camp_score,
                reference = scores.projection_score(
                    current_windows.features, id_geometry.u
                )
            )

        if z_semantic_score is None:
            final = semantic_score.logit.detach().to(torch.float64)
        elif z_camp_score is None:
            final = z_semantic_score
        else:
            final = scores.combine(
                z_semantic_score = z_semantic_score,
                z_camp_score = z_camp_score,
                semantic_weight = self.policy.fusion.semantic_weight
            )

        if self.recorder is not None:
            self.recorder.observe(
                vlm=vlm,
                semantic_score=semantic_score,
                window=self.window,
                current_windows=current_windows,
                id_geometry=id_geometry,
                camp_score=camp_score,
                z_camp_score=z_camp_score,
                z_semantic_score=z_semantic_score,
                final=final,
                diagnostic=diagnostic,
                latency_ms=(time.perf_counter() - started) * 1e3,
            )

        self.window.push(
            features=vlm.features,
            msp=vlm.msp,
            pseudo_class=vlm.pred,
            semantic_logit=semantic_score.logit,
        )

        return semantic_score.pred, final.float()


    def begin_diagnostic_stream(
        self,
        name: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:

        if self.recorder is not None:
            self.recorder.begin_stream(name, metadata)


    def end_diagnostic_stream(self) -> None:

        if self.recorder is not None:
            self.recorder.end_stream()


__all__ = ["CAMPPostprocessor"]
