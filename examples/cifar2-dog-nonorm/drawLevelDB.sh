#!/usr/bin/env sh
# This script draws leveldb elements into images

DB=./cifar2-leveldb-30/cifar-test-leveldb
TOOLS=../../build/tools
SAVEPATH=./cifar2-leveldb-30-images

echo "Drawing leveldb $DB"

rm -rf $SAVEPATH
mkdir $SAVEPATH

GLOG_logtostderr=1 $TOOLS/convert_leveldb_images.bin $DB $SAVEPATH

echo "Done."
