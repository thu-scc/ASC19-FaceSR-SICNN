#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs
set -e

EXAMPLE=/workspace/face_recognition/traindata/lmdb
DATA=/workspace/face_recognition/traindata
TOOLS=build/tools

TRAIN_DATA_ROOT=/workspace/face_recognition/


RESIZE_HEIGHT=0
RESIZE_WIDTH=0


echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    $TRAIN_DATA_ROOT \
    $DATA/multi_train_14x12_2_rand.txt \
    $EXAMPLE/multi_train_14x12_2_rand

echo "Done."
