#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/feedbacknet_visualize_neuron.bin alexnet_visualizer_neurons.prototxt 2>&1 |tee visualization.log
 
echo "Done."

