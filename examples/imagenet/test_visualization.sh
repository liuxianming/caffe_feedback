#!/usr/bin/env sh

TOOLS=../../build/tools

echo "Testing Visualization for ImageNet Images..."

GLOG_logtostderr=1 $TOOLS/test_convolutional_feedback.bin  \
    ./alexnet_visualization_feedback.prototxt \
    caffe_alexnet_model \
    2 fc8 1 
