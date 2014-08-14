#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/train_net.bin cifar2_full_solver.prototxt 2>&1 |tee feedback.log

echo "Done."
