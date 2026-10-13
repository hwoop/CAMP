from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from openood.utils.vlm.image.types import CachedImageFeatures


class BasePostprocessor:
    observes_provenance = False

    def __init__(
        self,
        config,
    ):
        self.config = config


    def setup(
        self,
        net: nn.Module,
        id_loader_dict,
        ood_loader_dict,
    ):
        pass


    def inference(
        self,
        net: nn.Module,
        data_loader: DataLoader,
        progress: bool = True,
        provenance: bool = False,
    ):

        pred_list, conf_list, label_list = [], [], []
        device = self._infer_device(net)
        observe = bool(provenance) and self.observes_provenance

        for batch in tqdm(data_loader, disable=not progress):
            postprocess_data = self.prepare_batch_data(batch, device=device)
            label = batch["label"].to(device)

            if observe:
                pred, conf = self.postprocess(net, postprocess_data, diagnostic=self._provenance(batch, label))
            else:
                pred, conf = self.postprocess(net, postprocess_data)

            pred_list.append(pred.cpu())
            conf_list.append(conf.cpu())
            label_list.append(label.cpu())

        # A recorder can be enabled without ground-truth/provenance export. Finish
        # that stream here too, instead of relying on the next stream or atexit.
        self.end_diagnostic_stream()

        if not pred_list:
            return (
                torch.empty(0, dtype=torch.long).numpy().astype(int),
                torch.empty(0, dtype=torch.float32).numpy(),
                torch.empty(0, dtype=torch.long).numpy().astype(int),
            )

        pred_list = torch.cat(pred_list).numpy().astype(int)
        conf_list = torch.cat(conf_list).numpy()
        label_list = torch.cat(label_list).numpy().astype(int)

        return pred_list, conf_list, label_list


    @staticmethod
    def _provenance(
        batch: Mapping[str, Any],
        label: torch.Tensor,
    ) -> dict[str, Any]:

        is_ood = batch["is_ood"] if "is_ood" in batch else label < 0
        if not torch.is_tensor(is_ood):
            is_ood = torch.as_tensor(is_ood)

        return {
            "label": label,
            "is_ood": is_ood.to(device=label.device).long(),
            "source_dataset": batch.get("source_dataset", None),
            "source_index": batch.get("source_index", None),
        }


    def prepare_batch_data(
        self,
        batch: dict,
        *,
        device: torch.device,
    ) -> Any:

        data = batch["data"].to(device)

        if not self._cached_image_feature_flag(batch, "image_features_cached"):
            return data

        return CachedImageFeatures(
            values=data,
            normalized=self._cached_image_feature_flag(
                batch,
                "image_features_normalized",
            ),
        )


    @staticmethod
    def _cached_image_feature_flag(
        batch: dict,
        key: str,
    ) -> bool:

        value = batch.get(key, False)
        if torch.is_tensor(value):
            flags = value.bool()
            if bool(torch.all(flags)):
                return True

            if not bool(torch.any(flags)):
                return False

            raise ValueError(f"Mixed cached/raw image batch is not supported: {key}.")

        return bool(value)


    def reset_memory(self):
        pass


    def begin_diagnostic_stream(
        self,
        name: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Called by the evaluator before it scores a stream. Used for diagnostics only; does nothing by default."""
        pass


    def end_diagnostic_stream(self) -> None:
        """Called by the evaluator after it has scored a stream. Used for diagnostics only; does nothing by default."""
        pass


    def _infer_device(
        self,
        net: nn.Module,
    ):

        try:
            return next(net.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
