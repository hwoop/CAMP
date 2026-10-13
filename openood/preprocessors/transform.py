import torchvision.transforms as tvs_trans

normalization_dict = {
    'clip': [[0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]],
}

interpolation_modes = {
    'bicubic': tvs_trans.InterpolationMode.BICUBIC,
}


class Convert:
    def __init__(
        self,
        mode='RGB',
    ):
        self.mode = mode


    def __call__(
        self,
        image,
    ):
        return image.convert(self.mode)
