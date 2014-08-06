#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/train_net_feedback.bin alexnet_solver.prototxt


echo "Done."
