#!/usr/bin/env sh
# This script converts the cifar data into leveldb format.

EXAMPLES=../../build/examples/cifar2-dog
DATA=../../data/cifar10
TOOLS=../../build/tools

echo "Creating leveldb WITH NOISE..."

for i in 1 5 10 15 20 30 35 40 45 50
do
    DBPATH=./cifar2-leveldb-$i
    echo "Adding noise $i -> $DBPATH"
    rm -rf $DBPATH
    mkdir $DBPATH

    GLOG_logtostderr=1 $EXAMPLES/convert_cifar_data_noisy.bin $DATA $DBPATH $i
    echo "Computing image mean..."
    GLOG_logtostderr=1 $TOOLS/compute_image_mean.bin $DBPATH/cifar-train-leveldb  mean.binaryproto

done

echo "Done."
