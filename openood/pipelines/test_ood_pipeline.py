import time

from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network
from openood.postprocessors import get_postprocessor
from openood.utils import setup_logger
from openood.utils.vlm.image.policy import ImageFeatureCachePolicy


class TestOODPipeline:
    def __init__(
        self,
        config,
    ) -> None:
        self.config = config


    def run(self):

        setup_logger(self.config)

        id_loader_dict = get_dataloader(self.config)
        ood_loader_dict = get_ood_dataloader(self.config)

        net = get_network(self.config.network)
        evaluator = get_evaluator(self.config)
        postprocessor = get_postprocessor(self.config)
        image_cache_policy = ImageFeatureCachePolicy.from_config(self.config)
        if image_cache_policy.enabled:
            from openood.utils.vlm.image.manager import ImageFeatureCacheManager

            image_cache_manager = ImageFeatureCacheManager(
                config=self.config,
                policy=image_cache_policy,
            )

            id_loader_dict, ood_loader_dict = image_cache_manager.prepare(
                net=net,
                id_loader_dict=id_loader_dict,
                ood_loader_dict=ood_loader_dict,
                postprocessor=postprocessor,
                evaluator=evaluator,
            )

        postprocessor.setup(net, id_loader_dict, ood_loader_dict)

        print('\n', flush=True)
        print(u'─' * 70, flush=True)

        timer = time.time()
        evaluator.eval_ood(net, id_loader_dict, ood_loader_dict, postprocessor)
        print('Time used for eval_ood: {:.0f}s'.format(time.time() - timer))
        print('Completed!', flush=True)
