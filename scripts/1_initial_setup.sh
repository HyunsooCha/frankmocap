#!/bin/zsh
set -e

docker build ../ -t frankmocap:0.1
echo "[INFO] docker build finished"
docker run -d -it --gpus=all --shm-size=120G \
    -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix:1 \
    -v $HOME/GitHub/frankmocap:/root/GitHub/frankmocap \
    -v $HOME/GitHub/IMavatar:/root/GitHub/IMavatar \
    -v $HOME/GitHub/IMavatar/data:/root/GitHub/IMavatar/data \
    -v $HOME/GitHub/IMavatar/code/runs:/root/GitHub/IMavatar/code/runs \
    --name frankmocap \
    frankmocap:0.1
echo "[INFO] docker run finished"
docker start frankmocap
echo "[INFO] docker start finished"
docker exec frankmocap sh scripts/install_frankmocap.sh
mkdir -p extra_data/smpl
cd ./extra_data/smpl/
wget https://www.dropbox.com/s/4xbm2cy65uxcxbb/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
wget https://www.dropbox.com/s/e8m5v88cd5lzz70/SMPLX_NEUTRAL.pkl
cd ../..
echo "[INFO] install_frankmocap finished"
docker attach --detach-keys "ctrl-z" frankmocap
