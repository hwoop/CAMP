from openood.utils import Config

from .ood_evaluator_clip_tta import OODEvaluatorClipTTA


def get_evaluator(
    config: Config,
):

    evaluators = {
        "ood_clip_tta": OODEvaluatorClipTTA,
    }
    return evaluators[config.evaluator.name](config)
