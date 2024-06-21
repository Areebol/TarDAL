import argparse
import logging
from pathlib import Path

import torch.backends.cudnn
import yaml

import scripts
from config import from_dict

if __name__ == '__main__':
    # args parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='/mnt/cephfs/home/areebol/sensors/TarDAL/config/tardal-train.yaml', help='config file path')
    parser.add_argument('--auth', default='aecdc69b1a817efc605df2d5be9dd7face113d04', help='wandb auth api key')
    args = parser.parse_args()

    # init config
    config = yaml.safe_load(Path(args.cfg).open('r'))
    config = from_dict(config)  # convert dict to object
    config = config

    # init logger
    log_f = '%(asctime)s | %(filename)s[line:%(lineno)d] | %(levelname)s | %(message)s'
    logging.basicConfig(level=config.debug.log, format=log_f)

    # init device & anomaly detector
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)

    # choose train script
    train_p = getattr(scripts, 'TrainF')

    # create script instance
    train = train_p(config, wandb_key=args.auth)
    train.run()
