from openood.utils import Config

from .test_preprocessor import TestStandardPreProcessor


def get_preprocessor(
    config: Config,
    split,
):
    return TestStandardPreProcessor(config)
