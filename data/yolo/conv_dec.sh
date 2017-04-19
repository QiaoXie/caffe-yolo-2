#!/usr/bin/env sh

CAFFE_ROOT=./

RESIZE_W=360
RESIZE_H=360

# 2007 + 2012 trainval
LIST_FILE="/data/wuliang/dec_data/train_voc_coco_np.txt"
LMDB_DIR="/data5/wuliang/lmdb/yolo_train_voc_coco_240_lmdb-1"
SHUFFLE=true

$CAFFE_ROOT/build/tools/convert_dec_data --resize_width=$RESIZE_W --resize_height=$RESIZE_H \
	$LIST_FILE $LMDB_DIR --encoded=true --encode_type=jpg --shuffle=$SHUFFLE

# 2007 test
LIST_FILE="/data/wuliang/Data/voc_data_1/2007_test.txt"
LMDB_DIR="/data5/wuliang/lmdb/yolo_test_voc_coco_240_lmdb-1"
SHUFFLE=false

$CAFFE_ROOT/build/tools/convert_dec_data --resize_width=$RESIZE_W --resize_height=$RESIZE_H \
	$LIST_FILE $LMDB_DIR --encoded=true --encode_type=jpg --shuffle=$SHUFFLE

