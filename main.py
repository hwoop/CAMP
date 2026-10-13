from openood.pipelines import get_pipeline
from openood.utils import configure_determinism_environment, seed_everything, setup_config


def main(config):
    seed_everything(int(config.seed))
    pipeline = get_pipeline(config)
    pipeline.run()


if __name__ == "__main__":
    config = setup_config()
    configure_determinism_environment(int(config.seed))
    main(config)
