import logging
from pathlib import Path

import torch
import yaml
from kornia.color import ycbcr_to_rgb
from torch.utils.data import DataLoader
from tqdm import tqdm

import time
from kornia.metrics import AverageMeter
import loader
from config import ConfigDict, from_dict
from pipeline.fuse import Fuse
from tools.dict_to_device import dict_to_device
from kornia.losses import MS_SSIMLoss, ssim_loss


class InferF:
    def __init__(self, config: str | Path | ConfigDict, save_dir: str | Path):
        # init logger
        log_f = '%(asctime)s | %(filename)s[line:%(lineno)d] | %(levelname)s | %(message)s'
        logging.basicConfig(level='INFO', format=log_f)
        logging.info(f'TarDAL-v1 Inference Script')

        # init config
        if isinstance(config, str) or isinstance(config, Path):
            config = yaml.safe_load(Path(config).open('r'))
            config = from_dict(config)  # convert dict to object
        else:
            config = config
        self.config = config

        # debug mode
        if config.debug.fast_run:
            logging.warning('fast run mode is on, only for debug!')

        # create save(output) folder
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f'create save folder {str(save_dir)}')
        self.save_dir = save_dir

        # init dataset & dataloader
        data_t = getattr(loader, config.dataset.name)  # dataset type
        self.data_t = data_t
        p_dataset = data_t(root=config.dataset.root, mode='pred', config=config)
        self.p_loader = DataLoader(
            p_dataset, batch_size=config.inference.batch_size, shuffle=False,
            collate_fn=data_t.collate_fn, pin_memory=True, num_workers=config.inference.num_workers,
        )

        # init pipeline
        fuse = Fuse(config, mode='inference')
        self.fuse = fuse

    @torch.inference_mode()
    def run(self):
        p_l = tqdm(self.p_loader, total=len(self.p_loader), ncols=120)
        inference_time = AverageMeter()
        for sample in p_l:
            input_shape = sample['shape']
            sample = dict_to_device(sample, self.fuse.device)
            start_time = time.time()
            # f_net forward
            fus = self.fuse.inference(ir=sample['ir'], vi=sample['vi'])
            inference_time.update(time.time() - start_time)
            # ssim loss
            ir_ssim = ssim_loss(fus, sample['ir'], window_size=11)
            vi_ssim = ssim_loss(fus, sample['vi'], window_size=11)
            text = f"ssim/ir:{ir_ssim:.6f} ssim/vi:{vi_ssim:.6f}"
            print(f"{sample['name']} {text}")
            
            # recolor
            if self.data_t.color and self.config.inference.grayscale is False:
                fus = torch.cat([fus, sample['cbcr']], dim=1)
                fus = ycbcr_to_rgb(fus)
            # save images
            self.data_t.pred_save(
                fus, [self.save_dir / name for name in sample['name']],
                shape=sample['shape'],
            )
            # save mask images
            if self.config.fuse.mask:
                mask = self.fuse.generator.mask(sample['ir'],sample['vi'])
                ir_mask_mean_value = torch.mean(mask[:,0,:,:])
                vi_mask_mean_value = torch.mean(mask[:,1,:,:])
                self.data_t.pred_save(
                mask[:,0,:,:], [self.save_dir / f"ir_mask_{name}" for name in sample['name']],
                shape=sample['shape'],
                )
                self.data_t.pred_save(
                mask[:,0,:,:], [self.save_dir / f"vi_mask_{name}" for name in sample['name']],
                shape=sample['shape'],
                )
        print(f'shape of sample : {input_shape}')
        print(f'elapsed time for running {len(self.p_loader)} samples : {inference_time.avg}s')