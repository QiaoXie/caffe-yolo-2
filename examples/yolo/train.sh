#!/usr/bin/env sh

CAFFE_HOME=../..

SOLVER=./gnet_solver.prototxt
WEIGHTS=../../models/bvlc_googlenet/bvlc_googlenet.caffemodel

$CAFFE_HOME/build/tools/caffe train \
    --solver=$SOLVER --weights=$WEIGHTS --gpu=4,5

