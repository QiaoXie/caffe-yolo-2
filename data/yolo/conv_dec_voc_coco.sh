#!/usr/bin/env sh

CAFFE_ROOT=./

RESIZE_W=112
RESIZE_H=112

# 2007 + 2012 trainval
LIST_FILE="/media/wuliang/Data/voc-data/voc_list/voc_train.txt"
LMDB_DIR="/media/wuliang/Data/lmdb/train_voc_all_112_lmdb-1"
SHUFFLE=true

$CAFFE_ROOT/build/tools/convert_dec_data --resize_width=$RESIZE_W --resize_height=$RESIZE_H \
	$LIST_FILE $LMDB_DIR --encoded=true --encode_type=jpg --shuffle=$SHUFFLE

# 2007 test
LIST_FILE="/media/wuliang/Data/voc-data/voc_list/2007_test.txt"
LMDB_DIR="/media/wuliang/Data/lmdb/test_voc_2007_112_lmdb-1"
SHUFFLE=false

$CAFFE_ROOT/build/tools/convert_dec_data --resize_width=$RESIZE_W --resize_height=$RESIZE_H \
	$LIST_FILE $LMDB_DIR --encoded=true --encode_type=jpg --shuffle=$SHUFFLE

