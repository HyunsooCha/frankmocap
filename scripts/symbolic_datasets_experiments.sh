#!/bin/zsh
rm $HOME/GitHub/frankmocap/mocap_output
set -e
# Get the hostname
hostname=$(hostname)

# Check if hostname is UltraFusion
if [[ "$hostname" == "ultrafusion" ]]; then
  echo "Hostname is ultrafusion"
  ln -s $HOME/Drive/Data/frankmocap/mocap_output $HOME/GitHub/frankmocap/mocap_output
elif [[ "$hostname" == "vclabdesktop" ]]; then
  echo "Hostname is vclabdesktop"
  ln -s $HOME/Drive/Data/frankmocap/mocap_output $HOME/GitHub/frankmocap/mocap_output
# elif [[ "$hostname" == "vclabserver2" ]]; then
#   echo "Hostname is vclabserver2"
#   ln -s $HOME/Dropbox/Data/IMavatar/data/original $HOME/Data/IMavatar/data/datasets/original
#   ln -s $HOME/Data/IMavatar/data/datasets $HOME/GitHub/IMavatar/data/datasets
#   ln -s $HOME/Dropbox/Data/IMavatar/data/experiments $HOME/GitHub/IMavatar/data/experiments
#   ln -s $HOME/Dropbox/Data/IMavatar/data/pretrain $HOME/GitHub/IMavatar/data/pretrain
# elif [[ "$hostname" == "vclabserver5" ]]; then
#   echo "Hostname is vclabserver5"
#   ln -s $HOME/Dropbox/Data/IMavatar/data/original $HOME/Data/IMavatar/data/datasets/original
#   ln -s $HOME/Data/IMavatar/data/datasets $HOME/GitHub/IMavatar/data/datasets
#   ln -s $HOME/Dropbox/Data/IMavatar/data/experiments $HOME/GitHub/IMavatar/data/experiments
#   ln -s $HOME/Dropbox/Data/IMavatar/data/pretrain $HOME/GitHub/IMavatar/data/pretrain
fi

