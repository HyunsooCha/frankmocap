#!/bin/zsh
conda remove -y -n frankmocap --all
set -e

source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate

echo "[INFO] create anaconda environment"
conda create -y -n frankmocap python=3.8
conda activate frankmocap

echo "[INFO] install pytorch"
# conda install -y pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

sudo apt install -y gcc
sudo apt install -y g++

echo "[INFO] install pytorch3D"
conda install -y -c fvcore -c iopath -c conda-forge fvcore iopath
pip install ninja
curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz
tar xzf 1.10.0.tar.gz
export CUB_HOME=$PWD/cub-1.10.0
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

echo "[INFO] install mesa etc"
sudo apt-get install -y libglu1-mesa libxi-dev libxmu-dev libglu1-mesa-dev freeglut3-dev libosmesa6-dev

echo "[INFO] install ffmpeg"
conda install -y ffmpeg

echo "[INFO] install requirements in frankmocap"
pip install -r docs/requirements.txt

echo "[INFO] install detectron2"
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

echo "[INFO] install frankmocap etc"
CUDA_VISIBLE_DEVICES=1 sh scripts/install_frankmocap.sh
# detectors/hand_object_detector가 예전 gpu에서만 install이 된다.

echo "[INFO] install scikit-learn etc"
pip install scikit-learn
sudo apt install -y xvfb
sudo add-apt-repository -y universe
sudo apt-get install -y pymol
echo "[INFO] finished"