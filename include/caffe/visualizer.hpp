// Copyright Xianming Liu

#ifndef CAFFE_VISUALIZER_HPP_
#define CAFFE_VISUALIZER_HPP_

#include <string>
#include <vector>

namespace caffe {
  template <typename Dtype>
  class Visualizer {
  public:
    explicit Visualizer(const VisualizerParameter& param);
    explicit Visualizer(const string& param_file);
    void Init(const VisualizerParameter& param);
    void Visualize();

  protected:
    VisualizerParameter param_;
    int iter_;
    shared_ptr<FeedbackNet<Dtype> > net_;

    DISABLE_COPY_AND_ASSIGN(Visualizer);
  };
}

#endif
