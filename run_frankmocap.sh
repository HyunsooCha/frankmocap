#!/bin/zsh
set -e

docker exec -e CUDA_VISIBLE_DEVICES=$1 frankmocap ./xvfb-run-safe python -m demo.demo_handmocap --input_path $2 --out_dir $3 --view_type ego_centric --save_bbox_output
docker exec -e CUDA_VISIBLE_DEVICES=$1 frankmocap chmod -R 777 $3