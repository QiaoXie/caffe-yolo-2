#!/usr/bin/env sh

CAFFE_HOME=../..

SOLVER=./tiny_solver.prototxt
WEIGHTS=../../yolo_models/darknet_v1.9.caffemodel

$CAFFE_HOME/build/tools/caffe train \
    --solver=$SOLVER --weights=$WEIGHTS --gpu=1

