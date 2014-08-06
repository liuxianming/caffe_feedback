// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <vector>

#include "caffe/neuron_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

using std::max;

namespace caffe {

  template <typename Dtype>
  Dtype ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
				      vector<Blob<Dtype>*>* top) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = (*top)[0]->mutable_cpu_data();
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
      top_data[i] = max(bottom_data[i], Dtype(0));
    }
    return Dtype(0);
  }

  template <typename Dtype>
  void ReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
				      const bool propagate_down,
				      vector<Blob<Dtype>*>* bottom) {
    if (propagate_down) {
      const Dtype* bottom_data = (*bottom)[0]->cpu_data();
      const Dtype* top_diff = top[0]->cpu_diff();
      Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
      const int count = (*bottom)[0]->count();
      for (int i = 0; i < count; ++i) {
	bottom_diff[i] = top_diff[i] * (bottom_data[i] > 0);
      }
    }
  }

  template <typename Dtype>
  void ReLULayer<Dtype>::UpdateEqFilter(const Blob<Dtype>* top_filter, 
					const vector<Blob<Dtype>*>& input)
  {
    //Initialization:
    //The size of eq_filter_ is the same as top_filter
    this->eq_filter_->Reshape(top_filter->num(),
			      top_filter->channels(),
			      top_filter->height(),
			      top_filter->width());

    this->eq_filter_->CopyFrom(*top_filter);

    //add constraints:
    Dtype* eq_filter_data = this->eq_filter_->mutable_cpu_data();
    int inputsize = input[0]->count() / input[0]->num();
    const Dtype* input_data = input[0]->cpu_data();

    for (int m = 0; m<input[0]->num(); m++) {
        for (int offset = 0; offset<inputsize; offset++) {
            Dtype _value = *(input_data + input[0]->offset(m) + offset);
            if (_value <= 0){
                for(int c = 0; c<this->eq_filter_->channels(); c++){
                    for(int o = 0; o<this->eq_filter_->height(); o++){
                        *(eq_filter_data + offset + this->eq_filter_->offset(m, c, o)) 
			  = (Dtype) 0.;
                    } // each output
                } //each channel
            }
        }//for each input element
    }// for each image in the mini-batch
  }


  INSTANTIATE_CLASS(ReLULayer);


}  // namespace caffe
