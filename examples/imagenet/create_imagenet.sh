#!/usr/bin/env sh
# Create the imagenet leveldb inputs
# N.B. set the path to the imagenet train + val data dirs

TOOLS=../../build/tools
DATA=../../data/ilsvrc12

echo "Creating leveldb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset.bin \
    $DATA/train/ \
    $DATA/train.txt \
    imagenet_train_leveldb 1 255

GLOG_logtostderr=1 $TOOLS/convert_imageset.bin \
    $DATA/val/ \
    $DATA/val.txt \
    imagenet_val_leveldb 1 255

echo "Done."
