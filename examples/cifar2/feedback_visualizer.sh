#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/feedbacknet_visualize.bin cifar2_visualizer.prototxt 2>&1 |tee visualization.log
 
echo "Done."

