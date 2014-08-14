#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/train_net_feedback.bin cifar2_full_solver_feedback.prototxt 2>&1 |tee feedback.log

echo "Done."
