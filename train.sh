source /mnt/cephfs/home/areebol/miniconda3/bin/activate tardal
export CUDA_VISIBLE_DEVICES=6
# ROOT=/mnt/cephfs/home/areebol/sensors/TarDAL/config/tardal-train-wo-mask
ROOT=/mnt/cephfs/home/areebol/sensors/TarDAL/config/tardal-train-w-mask
# python train.py --cfg ${ROOT}/tardal-train-road.yaml 
# python train.py --cfg ${ROOT}/tardal-train-m3fd.yaml 
# python train.py --cfg ${ROOT}/tardal-train-llvip.yaml 

python train.py --cfg ${ROOT}/tardal-train-m3fd.yaml 
