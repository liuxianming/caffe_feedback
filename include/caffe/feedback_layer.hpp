// Copyright: Xianming Liu, @July 8, 2014
// The base class for feedback layers
// both convolutional layers, pooling layers, fully connected layers
// are override from this base class
// Add a virtual method UpdateEqFilter(),
// a data member eq_filter_ and corresponding properties

#ifndef CAFFE_FEEDBACK_LAYER_HPP_
#define CAFFE_FEEDBACK_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

using std::vector;

namespace caffe {

  template <typename Dtype>
  class FeedbackLayer : public Layer<Dtype>{

  public:

    explicit FeedbackLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
    // Returns the equivalent weights from current input to outputs:
    inline Blob<Dtype> >* eq_filter() {
      return eq_filter_;
    }
  protected:
    // The function to update the eq_filter_ given current layer and the filter of upper layers
    // top_filter: the eq_filter from the top layer
    // input: input from the bottom layer(s)
    virtual void UpdateEqFilter(const Blob<Dtype>* top_filter,
				const vector<Blob<Dtype>*>& input) = 0;

    // eq_filter_: The equivalent weights from current input to final output:
    // That is: final output = eq_filter_ * bottom
    // Size of eq_filter_
    // image_num * output_channels * output_size * input_size
    //      |              |               |             |
    //    (num)        (channel)        (height)      (width)
    shared_ptr<Blob<Dtype> > eq_filter_;
  };
} //namespace caffe

#endif //CAFFE_FEEDBACK_LAYER_HPP_
