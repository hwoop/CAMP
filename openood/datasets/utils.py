from torch.utils.data import DataLoader

from openood.preprocessors.utils import get_preprocessor
from openood.utils.config import Config

from .imglist_dataset import ImglistDataset


def get_dataloader(
    config: Config,
):

    dataset_config = config.dataset
    dataloader_dict = {}

    for split in dataset_config.split_names:
        split_config = dataset_config[split]
        preprocessor = get_preprocessor(config=config, split=split)
        dataset = ImglistDataset(
            name=dataset_config.name + '_' + split,
            imglist_pth=split_config.imglist_pth,
            data_dir=split_config.data_dir,
            num_classes=dataset_config.num_classes,
            preprocessor=preprocessor,
            data_aux_preprocessor=preprocessor
        )

        dataloader = DataLoader(
            dataset,
            batch_size=split_config.batch_size,
            shuffle=split_config.shuffle,
            num_workers=dataset_config.num_workers
        )

        dataloader_dict[split] = dataloader

    return dataloader_dict


def get_ood_dataloader(
    config: Config,
):

    ood_config = config.ood_dataset
    dataloader_dict = {}

    for split in ood_config.split_names:
        split_config = ood_config[split]
        preprocessor = get_preprocessor(config=config, split=split)
        sub_dataloader_dict = {}

        for dataset_name in split_config.datasets:
            dataset_config = split_config[dataset_name]
            dataset = ImglistDataset(
                name=ood_config.name + '_' + split,
                imglist_pth=dataset_config.imglist_pth,
                data_dir=dataset_config.data_dir,
                num_classes=ood_config.num_classes,
                preprocessor=preprocessor,
                data_aux_preprocessor=preprocessor
            )

            dataloader = DataLoader(
                dataset,
                batch_size=ood_config.batch_size,
                shuffle=ood_config.shuffle,
                num_workers=ood_config.num_workers
            )

            sub_dataloader_dict[dataset_name] = dataloader

        dataloader_dict[split] = sub_dataloader_dict

    return dataloader_dict
