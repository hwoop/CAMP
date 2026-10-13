import logging
import os
import os.path as osp
import sys

import yaml


class Logger:
    """Imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py"""
    def __init__(
        self,
        fpath=None,
    ):

        self.console = sys.stdout
        self.file = None

        if fpath is not None:
            os.makedirs(osp.dirname(fpath), exist_ok=True)
            self.file = open(fpath, 'w')


    def __del__(self):
        self.close()


    def write(
        self,
        msg,
    ):

        self.console.write(msg)

        if self.file is not None:
            self.file.write(msg)


    def flush(self):

        self.console.flush()

        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())


    def close(self):

        self.console.close()

        if self.file is not None:
            self.file.close()


def setup_logger(
    config,
):

    print('------------------ Config --------------------------', flush=True)
    print(config, flush=True)
    print(u'─' * 70, flush=True)

    output = config.output_dir
    print('Output dir: {}'.format(output), flush=True)
    save_logger(config=config, output=output)


def save_logger(
    config,
    output,
):

    print('Output directory path: {}'.format(output), flush=True)
    os.makedirs(output, exist_ok=True)

    config_save_path = osp.join(output, 'config.yml')
    with open(config_save_path, 'w') as f:
        yaml.dump(data=config, stream=f, default_flow_style=False, sort_keys=False, indent=2)

    fpath = osp.join(output, 'log.txt')
    sys.stdout = Logger(fpath)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    for handler in root.handlers[:]:
        root.removeHandler(handler)

    sh = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')
    sh.setFormatter(formatter)
    root.addHandler(sh)

    logging.info("Python logging system is now integrated with sys.stdout Logger.")
