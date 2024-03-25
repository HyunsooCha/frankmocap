#!/bin/bash
set -e
# Copyright (c) Facebook, Inc. and its affiliates.

cd detectors

pip install gdown

# Install 100-DOH hand-object detectors
# compile
cd hand_object_detector/lib
python setup.py build develop
cd ../../