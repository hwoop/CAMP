from openood.utils import Config

from .test_ood_pipeline import TestOODPipeline


def get_pipeline(
    config: Config,
):

    pipelines = {
        'test_ood': TestOODPipeline,
    }

    return pipelines[config.pipeline.name](config)
