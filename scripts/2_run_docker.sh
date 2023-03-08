#!/bin/zsh
docker rm frankmocap
set -e

docker run -d -it --gpus=all --shm-size=120G \
    -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix:1 \
    -v $HOME/GitHub/frankmocap:/root/GitHub/frankmocap \
    --name frankmocap \
    frankmocap:0.1