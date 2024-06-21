source /mnt/cephfs/home/areebol/miniconda3/bin/activate tardal
export CUDA_VISIBLE_DEVICES=6
DATA="road"
VERSION="4cfwm02x"

# wo mask 
python infer.py --cfg /mnt/cephfs/home/areebol/sensors/TarDAL/config/tardal-train-wo-mask/tardal-infer.yaml \
                --save_dir ./runs/wo-mask --corrupt True

# w mask
# python infer.py --cfg /mnt/cephfs/home/areebol/sensors/TarDAL/config/tardal-train-w-mask/tardal-infer.yaml \
#                 --save_dir ./runs/mask-c --corrupt True
