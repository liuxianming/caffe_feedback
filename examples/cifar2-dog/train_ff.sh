#!/usr/bin/env sh

TOOLS=../../build/tools

LOG=train_ff.log
rm -f $LOG
GLOG_logtostderr=1 $TOOLS/train_net.bin cifar2_ff_solver.prototxt 2>&1 |tee $LOG

echo "Done."
