#!/usr/bin/env sh

TOOLS=../../build/tools

LOG=visualize_fb_ff.log
rm -f $LOG
GLOG_logtostderr=1 $TOOLS/feedbacknet_visualize.bin cifar2_fb_ff_visualizer.prototxt 2>&1 | tee $LOG

echo "Done."
