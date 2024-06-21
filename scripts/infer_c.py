import logging
from pathlib import Path

import wandb
import torch
import yaml
from kornia.color import ycbcr_to_rgb
from kornia.metrics import AverageMeter
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

import loader
from config import ConfigDict, from_dict
from pipeline.fuse import Fuse
from tools.dict_to_device import dict_to_device
from kornia.losses import MS_SSIMLoss, ssim_loss
from loader.utils.reader import img_text

class InferC:
    def __init__(self, config: str | Path | ConfigDict, save_dir: str | Path, wandb_key: str=None):
        # init logger
        log_f = '%(asctime)s | %(filename)s[line:%(lineno)d] | %(levelname)s | %(message)s'
        logging.basicConfig(level='INFO', format=log_f)
        logging.info(f'TarDAL-v1 Inference Corruption Script')

        # init config
        if isinstance(config, str) or isinstance(config, Path):
            config = yaml.safe_load(Path(config).open('r'))
            config = from_dict(config)  # convert dict to object
        else:
            config = config
        self.config = config

        # wandb run
        wandb.login(key=wandb_key)  # wandb api key
        runs = wandb.init(project='TarDAL', config=config, mode=config.debug.wandb_mode)
        self.runs = runs

        # debug mode
        if config.debug.fast_run:
            logging.warning('fast run mode is on, only for debug!')

        # create save(output) folder
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f'create save folder {str(save_dir)}')
        self.save_dir = save_dir

        methods = ['Brightness','Snow','Fog','Contrast']
        self.p_loaders = []
        for method in methods:
            for severity_level in range(1,6):
                # init dataset & dataloader
                data_t = getattr(loader, config.dataset.name)  # dataset type
                self.data_t = data_t
                p_dataset = data_t(root=config.dataset.root, mode='corrupt', config=config, 
                                   method=method,severity_level=severity_level)
                p_loader = DataLoader(
                    p_dataset, batch_size=config.inference.batch_size, shuffle=False,
                    collate_fn=data_t.collate_fn, pin_memory=True, num_workers=config.inference.num_workers,
                )
                self.p_loaders.append((method,severity_level,p_loader))

        # init pipeline
        fuse = Fuse(config, mode='inference')
        self.fuse = fuse

    @torch.inference_mode()
    def run(self):
        for method,severity_level,p_loader in self.p_loaders:
            p_l = tqdm(p_loader, total=len(p_loader), ncols=120)
            log_dict = {}
            ir_masks, vi_masks, ir_o_masks, vi_o_masks, fuss, fuss_o = [],[],[],[],[],[]
            imgs, imgs_o = [], []
            ir_mean, vi_mean, ir_o_mean, vi_o_mean = {AverageMeter() for _ in range(4)}
            for sample in p_l:
                sample = dict_to_device(sample, self.fuse.device)
                imgs.append(sample['vi'])
                imgs_o.append(sample['o_vi'])
                # f_net forward
                fus = self.fuse.inference(ir=sample['ir'], vi=sample['vi'])
                # ssim loss
                ir_ssim = ssim_loss(fus, sample['ir'], window_size=11)
                vi_ssim = ssim_loss(fus, sample['o_vi'], window_size=11)
                text = f"ssim/ir:{ir_ssim:.6f} ssim/vi:{vi_ssim:.6f}"
                logging.info(f"{sample['name']} {text}")
                
                # recolor
                if self.data_t.color and self.config.inference.grayscale is False:
                    fus = torch.cat([fus, sample['cbcr']], dim=1)
                    fus = ycbcr_to_rgb(fus)
                # log images
                fus = img_text(fus,text=text)
                fuss.append(fus)
                
                # f_net original forward
                fus_o = self.fuse.inference(ir=sample['ir'], vi=sample['o_vi'])
                # ssim loss
                ir_o_ssim = ssim_loss(fus_o, sample['ir'], window_size=11)
                vi_o_ssim = ssim_loss(fus_o, sample['vi'], window_size=11)
                text = f"ssim/ir:{ir_o_ssim:.6f} ssim/vi:{vi_o_ssim:.6f}"
                logging.info(f"{sample['name']} {text}")
                
                # recolor
                if self.data_t.color and self.config.inference.grayscale is False:
                    fus_o = torch.cat([fus_o, sample['o_cbcr']], dim=1)
                    fus_o = ycbcr_to_rgb(fus_o)
                # log images
                fus_o = img_text(fus_o,text=text)
                fuss_o.append(fus_o)
                
                # log mask images
                if self.config.fuse.mask:
                    # corrupt mask
                    mask = self.fuse.generator.mask(sample['ir'],sample['vi'])
                    ir_mask_mean_value = torch.mean(mask[:,0,:,:])
                    vi_mask_mean_value = torch.mean(mask[:,1,:,:])
                    ir_mean.update(ir_mask_mean_value.item())
                    vi_mean.update(vi_mask_mean_value.item())
                    # ir_mask = img_text(mask[:,0,:,:],text=f"ir_mask_mean_value:{ir_mask_mean_value:.6f}")
                    ir_masks.append(img_text(mask[:,0,:,:],text=f"ir_mask_mean_value:{ir_mask_mean_value:.6f}"))
                    vi_masks.append(img_text(mask[:,0,:,:],text=f"vi_mask_mean_value:{vi_mask_mean_value:.6f}"))
                    
                    # original mask
                    o_mask = self.fuse.generator.mask(sample['ir'],sample['o_vi'])
                    ir_o_mask_mean_value = torch.mean(o_mask[:,0,:,:])
                    vi_o_mask_mean_value = torch.mean(o_mask[:,1,:,:])
                    ir_o_mean.update(ir_o_mask_mean_value.item())
                    vi_o_mean.update(vi_o_mask_mean_value.item())
                    ir_o_masks.append(img_text(o_mask[:,0,:,:],text=f"ir_o_mask_mean_value:{ir_o_mask_mean_value:.6f}"))
                    vi_o_masks.append(img_text(o_mask[:,0,:,:],text=f"vi_o_mask_mean_value:{vi_o_mask_mean_value:.6f}"))
            if self.config.fuse.mask:
                log_dict |= {
                            f'{method}/ir_mask': wandb.Image(np.concatenate(ir_masks)),
                            f'{method}/ir_o_mask': wandb.Image(np.concatenate(ir_o_masks)),
                            f'{method}/vi_mask': wandb.Image(np.concatenate(vi_masks)),
                            f'{method}/vi_o_mask': wandb.Image(np.concatenate(vi_o_masks)),
                            f'{method}/ir_mean': ir_mean.avg,
                            f'{method}/vi_mean': vi_mean.avg,
                            f'{method}/ir_o_mean': ir_o_mean.avg,
                            f'{method}/vi_o_mean': vi_o_mean.avg,
                            }
            log_dict |= {
                        f'{method}/fus': wandb.Image(np.concatenate(fuss)),
                        f'{method}/fus_o': wandb.Image(np.concatenate(fuss_o)),
                        f'{method}/vi': wandb.Image(torch.concatenate(imgs)),
                        f'{method}/vi_o': wandb.Image(torch.concatenate(imgs_o)),
                        "severity_level": severity_level
                        }
            # update wandb
            self.runs.log(log_dict)