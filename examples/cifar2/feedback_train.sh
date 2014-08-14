#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/finetune_net_feedback.bin cifar10_full_solver_feedback.prototxt cifar10_full_iter_70000 2>&1 |tee feedback.log

echo "Done."
