#!/usr/bin/env sh

TOOLS=../../build/tools

LOG=train_fb.log

rm -f $LOG
GLOG_logtostderr=1 $TOOLS/train_net_feedback.bin cifar2_fb_solver.prototxt 2>&1 |tee $LOG

echo "Done."
