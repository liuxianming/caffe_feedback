// Copyright 2014 Xianming Liu
//
// This is a simple script that allows one to quickly train a network whose
// parameters are specified by text format protocol buffers.
// Usage:
//    feedbacknet_visualize visualize_proto_file 

#include <cuda_runtime.h>

#include <cstring>

#include "caffe/caffe.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  //project all the output to standard output
  ::google::LogToStderr();
  if (argc != 2) {
    LOG(ERROR) << "Usage: feedbacknet_visualize visualize_proto_file";
    return 1;
  }

  VisualizerParameter visualizer_param;
  ReadProtoFromTextFileOrDie(argv[1], &visualizer_param);

  LOG(INFO) << "Read task from "<<argv[1];
  Visualizer<float> visualizer(visualizer_param);
  visualizer.Visualize();
  LOG(INFO) << "Visualization Done.";

  return 0;
}
