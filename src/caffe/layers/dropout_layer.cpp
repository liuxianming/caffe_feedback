// Copyright 2014 BVLC and contributors.

// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.

#include <vector>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

  template <typename Dtype>
  void DropoutLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
				  vector<Blob<Dtype>*>* top) {
    NeuronLayer<Dtype>::SetUp(bottom, top);
    // Set up the cache for random number generation
    rand_vec_.reset(new SyncedMemory(bottom[0]->count() * sizeof(int)));
    threshold_ = this->layer_param_.dropout_param().dropout_ratio();
    DCHECK(threshold_ > 0.);
    DCHECK(threshold_ < 1.);
    scale_ = 1. / (1. - threshold_);
    uint_thres_ = static_cast<unsigned int>(UINT_MAX * threshold_);
    //Setup EqFilter
    this->eq_filter_ = new Blob<Dtype>(bottom[0]->num(), 1, 1, bottom[0]->channels() * bottom[0]->height() * bottom[0]->width());
  }

  template <typename Dtype>
  Dtype DropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
					 vector<Blob<Dtype>*>* top) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = (*top)[0]->mutable_cpu_data();
    int* mask = reinterpret_cast<int*>(rand_vec_->mutable_cpu_data());
    const int count = bottom[0]->count();
    if (Caffe::phase() == Caffe::TRAIN) {
      // Create random numbers
      caffe_rng_bernoulli(count, 1. - threshold_, mask);
      for (int i = 0; i < count; ++i) {
	top_data[i] = bottom_data[i] * mask[i] * scale_;
      }
    } else {
      caffe_copy(bottom[0]->count(), bottom_data, top_data);
    }
    return Dtype(0);
  }

  template <typename Dtype>
  void DropoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
					 const bool propagate_down,
					 vector<Blob<Dtype>*>* bottom) {
    CHECK(Caffe::phase() == Caffe::TRAIN);
    if (propagate_down) {
      const Dtype* top_diff = top[0]->cpu_diff();
      Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
      const int* mask = reinterpret_cast<const int*>(rand_vec_->cpu_data());
      const int count = (*bottom)[0]->count();
      for (int i = 0; i < count; ++i) {
	bottom_diff[i] = top_diff[i] * mask[i] * scale_;
      }
    }
  }

  template<typename Dtype>
  void DropoutLayer<Dtype>::UpdateEqFilter(const Blob<Dtype>* top_filter,
			      const vector<Blob<Dtype>*>& input) {
    //copy the eq_filter from input to this->eq_filter_
    this->eq_filter_->CopyFrom( *top_filter, false, true);
  }


  INSTANTIATE_CLASS(DropoutLayer);


}  // namespace caffe
