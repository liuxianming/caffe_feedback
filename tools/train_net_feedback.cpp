// Copyright 2014 Xianming Liu
//
// Train net using feedback neural networks
// Usage:
//    train_net_feedback net_proto_file solver_proto_file [resume_point_file]

#include <cuda_runtime.h>

#include <cstring>

#include "caffe/caffe.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  //project all the output to standard output
  ::google::LogToStderr();
  if (argc < 2 || argc > 3) {
    LOG(ERROR) << "Usage: train_net solver_proto_file [resume_point_file]";
    return 1;
  }

  SolverParameter solver_param;
  ReadProtoFromTextFileOrDie(argv[1], &solver_param);

  LOG(INFO) << "Starting Optimization";
  SGDFeedbackSolver<float> solver(solver_param);
  if (argc == 3) {
    LOG(INFO) << "Resuming from " << argv[2];
    solver.Solve(argv[2]);
  } else {
    solver.Solve();
  }
  LOG(INFO) << "Optimization Done.";

  return 0;
}
