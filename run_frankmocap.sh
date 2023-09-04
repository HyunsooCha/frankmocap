#!/bin/zsh
set -e

docker start frankmocap
docker attach --detach-keys "ctrl-z" frankmocap

docker exec -e CUDA_VISIBLE_DEVICES=0 frankmocap ./xvfb-run-safe python -m demo.demo_handmocap --input_path $1 --out_dir $2 --view_type ego_centric --save_bbox_output