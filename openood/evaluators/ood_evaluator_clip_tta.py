import csv
import os
from typing import Dict, Mapping

import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from openood.postprocessors import BasePostprocessor
from openood.utils import Config
from openood.utils.config import get_config
from openood.utils.reproducibility import make_torch_generator, seed_worker

from .metrics import compute_all_metrics
from .stream_protocol import StreamProtocol, build_stream_plans


class OODEvaluatorClipTTA:
    # read by ImageFeatureCacheManager
    supports_image_feature_cache = True
    image_feature_cache_required_id_splits = {"test"}
    image_feature_cache_required_ood_splits = {"nearood", "farood"}

    def __init__(
        self,
        config: Config,
    ):

        self.config = config
        self._ood_csv_initialized = False
        self._stream_files_initialized = False


    def eval_ood(
        self,
        net: nn.Module,
        id_loader_dict: Dict[str, DataLoader],
        ood_loader_dict: Dict[str, Dict[str, DataLoader]],
        postprocessor: BasePostprocessor,
    ):

        self._ood_csv_initialized = False
        self._stream_files_initialized = False
        protocol = StreamProtocol.from_config(get_config(obj=self.config.evaluator, name='stream', default=None))
        # diagnostic.enabled (configs/diagnostics/sample_provenance.yml): hand each batch's ground truth and source
        # row to a postprocessor that records diagnostics; the scores do not change (observation only)
        self._provenance = bool(get_config(obj=get_config(obj=self.config, name='diagnostic', default=None), name='enabled', default=False))

        net.eval()

        assert 'test' in id_loader_dict, 'id_data_loaders should have the key: test!'
        id_loader = id_loader_dict['test']

        print(f'Performing inference on {self.config.dataset.name} dataset...', flush=True)

        if protocol.composition == 'mixed_groups':
            print(u'─' * 70, flush=True)
            self._eval_planned(
                net,
                id_loader,
                ood_loader_dict,
                postprocessor,
                splits=('nearood', 'farood'),
                protocol=protocol
            )

            return

        for split in ('nearood', 'farood'):
            print(u'─' * 70, flush=True)
            loaders = self._leaf_loaders(ood_loader_dict, split)
            if loaders is None:
                continue

            print(f'Processing {split}...', flush=True)

            if protocol.is_default:
                self._eval_benchmark(
                    net,
                    id_loader,
                    loaders,
                    postprocessor,
                    split=split
                )
            else:
                self._eval_planned(
                    net,
                    id_loader,
                    ood_loader_dict,
                    postprocessor,
                    splits=(split,),
                    protocol=protocol
                )


    @staticmethod
    def _leaf_loaders(
        ood_loader_dict: Dict[str, Dict[str, DataLoader]],
        split: str,
    ):

        if split not in ood_loader_dict:
            return None

        loaders = ood_loader_dict[split]
        if not isinstance(loaders, Mapping) or not loaders:
            raise ValueError(
                f"OOD {split} target split must be a non-empty mapping."
            )

        return loaders


    def _eval_benchmark(
        self,
        net: nn.Module,
        id_loader: DataLoader,
        loaders: Mapping[str, DataLoader],
        postprocessor: BasePostprocessor,
        *,
        split: str,
    ) -> None:

        metrics_list = []
        stream_seed = int(self.config.seed)

        for dataset_name, ood_dl in loaders.items():
            print(f'Performing inference on {dataset_name} dataset...', flush=True)

            postprocessor.reset_memory()
            postprocessor.setup(net, id_loader, ood_dl)
            postprocessor.begin_diagnostic_stream(dataset_name, {'group': split, 'composition': 'per_leaf'})

            combined_dataset = ConcatDataset([ood_dl.dataset, id_loader.dataset])
            print(
                f'Generating combined dataset with ID and OOD dataset of {dataset_name}, '
                f'total size {len(combined_dataset)}, stream seed {stream_seed}'
            )

            combined_dataloader = DataLoader(
                combined_dataset,
                batch_size=id_loader.batch_size,
                num_workers=id_loader.num_workers,
                shuffle=True,
                generator=make_torch_generator(stream_seed),
                worker_init_fn=seed_worker,
            )

            pred, conf, label = postprocessor.inference(
                net=net,
                data_loader=combined_dataloader,
                provenance=self._provenance
            )

            self._save_scores(pred, conf, label, dataset_name)

            print(f'Computing metrics on {dataset_name} dataset...')
            ood_metrics = compute_all_metrics(conf=conf, label=label, pred=pred)
            self._save_csv(ood_metrics, dataset_name=dataset_name)
            metrics_list.append(ood_metrics)

        self._save_mean_csv(metrics_list, split)


    def _eval_planned(
        self,
        net: nn.Module,
        id_loader: DataLoader,
        ood_loader_dict: Dict[str, Dict[str, DataLoader]],
        postprocessor: BasePostprocessor,
        *,
        splits,
        protocol: StreamProtocol,
    ) -> None:

        groups = {}

        for split in splits:
            loaders = self._leaf_loaders(ood_loader_dict, split)
            if loaders is not None:
                groups[split] = {name: loader.dataset for name, loader in loaders.items()}

        if not groups:
            return

        seed = int(self.config.seed)
        plans = build_stream_plans(
            protocol,
            id_dataset=id_loader.dataset,
            groups=groups,
            seed=seed
        )

        results = {}

        for plan in plans:
            print(
                f'Performing inference on stream {plan.name} [{protocol.describe()}], '
                f'total size {len(plan)}, stream seed {seed}',
                flush=True,
            )

            postprocessor.reset_memory()
            postprocessor.setup(
                net,
                id_loader,
                DataLoader(plan.ood_dataset(), batch_size=id_loader.batch_size)
            )

            postprocessor.begin_diagnostic_stream(
                plan.name,
                {
                    'composition': plan.composition,
                    'protocol': protocol.describe()
                }
            )

            stream_loader = DataLoader(
                plan.dataset,
                batch_size=id_loader.batch_size,
                sampler=plan.order.tolist(),
                num_workers=id_loader.num_workers,
                worker_init_fn=seed_worker,
            )

            pred, conf, label = postprocessor.inference(
                net=net,
                data_loader=stream_loader,
                provenance=self._provenance
            )

            pred, conf, label = np.asarray(pred), np.asarray(conf), np.asarray(label)

            for index, part in plan.ood_parts():
                select = plan.evaluation_positions(index)
                self._save_scores(pred[select], conf[select], label[select], part.leaf)
                results[(part.group, part.leaf)] = compute_all_metrics(conf=conf[select], label=label[select], pred=pred[select])

            self._write_stream_tables(plan)

        for split, members in groups.items():
            metrics_list = []

            for leaf in members:
                print(f'Computing metrics on {leaf} dataset...')
                self._save_csv(results[(split, leaf)], dataset_name=leaf)
                metrics_list.append(results[(split, leaf)])

            self._save_mean_csv(metrics_list, split)


    def _write_stream_tables(
        self,
        plan,
    ) -> None:

        mode = 'a' if self._stream_files_initialized else 'w'

        for filename, rows in (('stream_plan.csv', plan.parts_table()), ('stream_phases.csv', plan.phases())):
            with open(os.path.join(self.config.output_dir, filename), mode, newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))

                if mode == 'w':
                    writer.writeheader()

                writer.writerows(rows)

        self._stream_files_initialized = True


    def _save_mean_csv(
        self,
        metrics_list,
        split: str,
    ) -> None:

        print('Computing mean metrics...', flush=True)
        self._save_csv(np.mean(np.array(metrics_list), axis=0), dataset_name=split)


    def _save_csv(
        self,
        metrics,
        dataset_name,
    ):

        [fpr, auroc, aupr_in, aupr_out, accuracy] = metrics
        write_content = {
            'dataset': dataset_name,
            'FPR@95': '{:.2f}'.format(100 * fpr),
            'AUROC': '{:.2f}'.format(100 * auroc),
            'AUPR_IN': '{:.2f}'.format(100 * aupr_in),
            'AUPR_OUT': '{:.2f}'.format(100 * aupr_out),
            'ACC': '{:.2f}'.format(100 * accuracy),
        }

        fieldnames = list(write_content.keys())
        print('FPR@95: {:.2f}, AUROC: {:.2f}'.format(100 * fpr, 100 * auroc), end=' ', flush=True)
        print('AUPR_IN: {:.2f}, AUPR_OUT: {:.2f}'.format(100 * aupr_in, 100 * aupr_out), flush=True)
        print('ACC: {:.2f}'.format(accuracy * 100), flush=True)
        print(u'─' * 70, flush=True)

        csv_path = os.path.join(self.config.output_dir, 'ood.csv')
        mode = 'a' if self._ood_csv_initialized else 'w'
        with open(csv_path, mode, newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if mode == 'w':
                writer.writeheader()

            writer.writerow(write_content)

        self._ood_csv_initialized = True


    def _save_scores(
        self,
        pred,
        conf,
        gt,
        save_name,
    ):

        save_dir = os.path.join(self.config.output_dir, 'scores')
        os.makedirs(save_dir, exist_ok=True)
        np.savez(os.path.join(save_dir, save_name), pred=pred, conf=conf, label=gt)
