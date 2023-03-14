#!/bin/zsh
set -e

docker start frankmocap
docker attach --detach-keys "ctrl-z" frankmocap