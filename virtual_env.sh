# create virtual environment
conda create -n tardal python=3.10
source /mnt/cephfs/home/areebol/miniconda3/bin/activate tardal
# select pytorch version yourself
# install tardal requirements
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
