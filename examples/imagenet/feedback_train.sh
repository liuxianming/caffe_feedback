#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/finetune_net_feedback.bin alexnet_feedback_solver.prototxt caffe_alexnet_model 2>&1 |tee feedback.log

echo "Done."
