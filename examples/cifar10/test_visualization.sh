#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/test_convolutional_feedback.bin  \
    cifar10_full_test.prototxt cifar10_full_iter_70000 mean.binaryproto
