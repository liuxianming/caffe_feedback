// Copyright 2014 Xianming Liu

#include <algorithm>
#include <vector>
#include <cstdlib>

#include "caffe/neuron_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

#define THRESHOLD (Dtype) 0.0

namespace caffe {

  template <typename Dtype>
  void ReLUPlusLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom, 
				   vector<Blob<Dtype>*>* top){
    CHECK_EQ(bottom.size(), 1) << "ReLUPlus Layer takes a single blob as input.";
    CHECK_EQ(top->size(), 1) << "ReLUPlus Layer takes a single blob as output.";
    //initialize activation
    /*
    LOG(INFO)<<"setting up ReLUPlus Layer";
    LOG(INFO)<<"size of relu_plus: "<<bottom[0]->num()<< " "<< bottom[0]->channels() 
	     <<" "<<bottom[0]->height()<<" "<<bottom[0]->width();
    */    
    this->activation_ = new Blob<Dtype>(bottom[0]->num(), bottom[0]->channels(), 
					bottom[0]->height(), bottom[0]->width());
    //this->activation_->CopyFrom(*(bottom[0]), false, true);
    Reset();
    if ((*top)[0] != bottom[0]) {
      (*top)[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
			 bottom[0]->height(), bottom[0]->width());
    }
    //Setting up the eq_filter
    this->eq_filter_ = new Blob<Dtype>(bottom[0]->num(), 1, 1, bottom[0]->channels() * bottom[0]->height() * bottom[0]->width());
  }

  template<typename Dtype>
  void ReLUPlusLayer<Dtype>::Reset(){
    for(int idx = 0; idx < this->activation_->count(); ++idx) {
      *(this->activation_->mutable_cpu_data() + idx) = (Dtype) 1.;
    }
  }

  template <typename Dtype>
  Dtype ReLUPlusLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, 
					  vector<Blob<Dtype>*>* top) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* activation_data = this->activation_->cpu_data();
    Dtype* top_data = (*top)[0]->mutable_cpu_data();
    const int count = bottom[0]->count();
    //Use activation to active each neuron, which remebers the feedback status
    for (int i = 0; i < count; ++i) {
      top_data[i] = max(bottom_data[i], Dtype(0));
    }
    //Apply activation
    caffe_mul(bottom[0]->count(), top_data, activation_data, top_data);
    return Dtype(0);
  }

  template <typename Dtype>
  Dtype ReLUPlusLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, 
					  vector<Blob<Dtype>*>* top) {
    Forward_cpu(bottom, top);
  }

  template <typename Dtype>
  void ReLUPlusLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
				      const bool propagate_down,
				      vector<Blob<Dtype>*>* bottom) {
    if (propagate_down) {
      const Dtype* bottom_data = (*bottom)[0]->cpu_data();
      const Dtype* top_diff = top[0]->cpu_diff();
      const Dtype* activation_data = this->activation_->cpu_data();
      Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
      const int count = (*bottom)[0]->count();
      for (int i = 0; i < count; ++i) {
	bottom_diff[i] = top_diff[i] * (bottom_data[i] > 0) * activation_data[i];
      }
    }
  }

  template <typename Dtype>
  void ReLUPlusLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
				      const bool propagate_down,
				      vector<Blob<Dtype>*>* bottom) {
    Backward_cpu(top, propagate_down, bottom);
  }

  template <typename Dtype>
  void ReLUPlusLayer<Dtype>::UpdateEqFilter(const Blob<Dtype>* top_filter, 
					    const vector<Blob<Dtype>*>& input) {
    //Initialization:
    //The size of eq_filter_ is the same as top_filter
    /*
    this->eq_filter_->Reshape(top_filter->num(),
			      top_filter->channels(),
			      top_filter->height(),
			      top_filter->width());
    */

    this->eq_filter_->CopyFrom(*top_filter, false, true);

    //add constraints:
    Dtype* eq_filter_data = this->eq_filter_->mutable_cpu_data();
    Dtype* activation_data = this->activation_->mutable_cpu_data();
    int inputsize = input[0]->count() / input[0]->num();
    const Dtype* input_data = input[0]->cpu_data();

    /*
    //calculate the average of w*x
    //1. apply current activation to eq_filter
    caffe_mul(this->eq_filter_->count(), eq_filter_data, activation_data, eq_filter_data);
    Dtype* abs_input_data = new Dtype[input[0]->count()];
    Dtype* abs_eq_filter_data = new Dtype[this->eq_filter_->count()];
    for(int i = 0; i<input[0]->count(); ++i){
      abs_input_data[i] = fabs(max(input_data[i], Dtype(0.)));
      abs_eq_filter_data[i] = fabs(eq_filter_data[i]);
    }
    //2. calcuate the average
    Dtype* m_response = new Dtype[input[0]->num()];
    for(int n = 0; n<input[0]->num(); ++n){
      m_response[n] = caffe_cpu_dot<Dtype>(inputsize, 
					   abs_eq_filter_data + this->eq_filter_->offset(n), 
					   abs_input_data + input[0]->offset(n))
	/ inputsize;
    }
    */
    
    for (int m = 0; m<input[0]->num(); m++) {
      for (int offset = 0; offset<inputsize; offset++) {
	Dtype _value = *(input_data + input[0]->offset(m) + offset);
	Dtype top_filter_value = *(top_filter->cpu_data() + 
				   top_filter->offset(m) + offset);
	if(_value <= 0){
	  *(eq_filter_data + offset + this->eq_filter_->offset(m)) 
	    = (Dtype) 0.;
	}
	
	//add a new version of activation function: abs(wx) >= THRESHOLD
	int img_idx = offset / inputsize;	
	//if(fabs(top_filter_value * _value) < (1.0 * m_response[img_idx])) {
	if((top_filter_value)<= THRESHOLD) {
	  //set activation
	  *(activation_data + offset + this->activation_->offset(m)) = (Dtype) 0.;
	}
      }//for each input element
    }// for each image in the mini-batch
    //Apply activation to eq_filter
    caffe_mul(this->eq_filter_->count(), eq_filter_data, activation_data, eq_filter_data);
  }

  INSTANTIATE_CLASS(ReLUPlusLayer);

} //namespace caffe
