// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <vector>

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
  void ReLULayer<Dtype>::UpdateEqFilter(const vector<Blob<Dtype>*>& top_filter, const vector<Blob<Dtype>*>& input)
  {
    //Initialization:
    //The size of eq_filter_ is the same as top_filter
    for(int i = 0;i<top_filter.size(); i++) {
      shared_ptr<Blob<Dtype> > eq_filter(new Blob<Dtype>(top_filter[i]->num(),
							 top_filter[i]->channels(),
							 top_filter[i]->height(),
							 top_filter[i]->width()));

      eq_filter->CopyFrom(*top_filter[i]);

      //add constraints:
      Dtype* eq_filter_data = eq_filter->mutable_cpu_data();
      int inputsize = input[i]->count() / input[i]->num();
      Dtype* input_data = input[i]->mutable_cpu_data();

      for (int m = 0; m<input[i]->num(); m++) {
	for (int offset = 0; offset<inputsize; offset++)
	  {
	    if (*(input_data + offset) < 0){
	      for(int c = 0; c<eq_filter->channels(); c++){
		for(int o = 0; o<eq_filter->height(); o++){
		  *(eq_filter_data + offset + eq_filter->offset(m, c, o)) = 0;
		} // each output
	      } //each channel
	    }
	  }//for each input element
      }// for each image in the mini-batch

      this->eq_filter_.push_back(eq_filter);
    }//for each input source (by default, = 1)
  }


  INSTANTIATE_CLASS(ReLULayer);


}  // namespace caffe
