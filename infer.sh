source /mnt/cephfs/home/areebol/miniconda3/bin/activate tardal
export CUDA_VISIBLE_DEVICES=6
DATA="road"
VERSION="4cfwm02x"

# wo mask 
# python infer.py --cfg /mnt/cephfs/home/areebol/sensors/TarDAL/config/tardal-train-wo-mask/tardal-infer.yaml \
#                 --save_dir ./runs/wo-mask

# w mask
python infer.py --cfg /mnt/cephfs/home/areebol/sensors/TarDAL/config/tardal-train-w-mask/tardal-infer-TNO.yaml \
                --save_dir ./runs/mask | tee logs/infer/TNO.log
python infer.py --cfg /mnt/cephfs/home/areebol/sensors/TarDAL/config/tardal-train-w-mask/tardal-infer-ROAD.yaml \
                --save_dir ./runs/mask | tee logs/infer/ROAD.log
python infer.py --cfg /mnt/cephfs/home/areebol/sensors/TarDAL/config/tardal-train-w-mask/tardal-infer-LLVIP.yaml \
                --save_dir ./runs/mask | tee logs/infer/LLVIP.log
python infer.py --cfg /mnt/cephfs/home/areebol/sensors/TarDAL/config/tardal-train-w-mask/tardal-infer-M3FD.yaml \
                --save_dir ./runs/mask | tee logs/infer/M3FD.log
