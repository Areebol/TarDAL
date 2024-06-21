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
    parser.add_argument('--cfg', default='/mnt/cephfs/home/areebol/sensors/TarDAL/config/tardal-infer.yaml', help='config file path')
    parser.add_argument('--save_dir', default='./runs/tmp', help='fusion result save folder')
    parser.add_argument('--auth', default='aecdc69b1a817efc605df2d5be9dd7face113d04', help='wandb auth api key')
    parser.add_argument('--corrupt', default=False, help='corruption prediction')
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

    # choose inference script
    logging.info(f'enter {config.strategy} inference mode')
    match config.strategy:
        case 'fuse':
            if args.corrupt:
                infer_p = getattr(scripts, 'InferC')
                # create script instance
                infer = infer_p(config, args.save_dir, args.auth)
            else:
                infer_p = getattr(scripts, 'InferF')
                # create script instance
                infer = infer_p(config, args.save_dir)
            # check pretrained weights
            if config.fuse.pretrained is None:
                logging.warning('no pretrained weights specified, use official pretrained weights')
                config.fuse.pretrained = 'https://github.com/JinyuanLiu-CV/TarDAL/releases/download/v1.0.0/tardal-dt.pth'
        case _:
            raise ValueError(f'unknown strategy: {config.strategy}')

    infer.run()
