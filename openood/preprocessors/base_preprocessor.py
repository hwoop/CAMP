from openood.utils.config import Config

from .transform import interpolation_modes, normalization_dict


class BasePreprocessor():
    def __init__(
        self,
        config: Config,
    ):

        self.pre_size = config.dataset.pre_size
        self.image_size = config.dataset.image_size
        self.interpolation = interpolation_modes[config.dataset.interpolation]
        normalization_type = config.dataset.normalization_type
        self.mean = normalization_dict[normalization_type][0]
        self.std = normalization_dict[normalization_type][1]


    def setup(
        self,
        **kwargs,
    ):
        pass


    def __call__(
        self,
        image,
    ):
        return self.transform(image)
