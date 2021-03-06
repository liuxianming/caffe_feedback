// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

#define GPUMODE true

namespace caffe {

  template <typename Dtype>
  void InnerProductLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
				       vector<Blob<Dtype>*>* top) {
    CHECK_EQ(bottom.size(), 1) << "IP Layer takes a single blob as input.";
    CHECK_EQ(top->size(), 1) << "IP Layer takes a single blob as output.";
    const int num_output = this->layer_param_.inner_product_param().num_output();
    bias_term_ = this->layer_param_.inner_product_param().bias_term();
    // Figure out the dimensions
    M_ = bottom[0]->num();
    K_ = bottom[0]->count() / bottom[0]->num();
    N_ = num_output;
    (*top)[0]->Reshape(bottom[0]->num(), num_output, 1, 1);
    // Check if we need to set up the weights
    if (this->blobs_.size() > 0) {
      LOG(INFO) << "Skipping parameter initialization";
    } else {
      if (bias_term_) {
	this->blobs_.resize(2);
      } else {
	this->blobs_.resize(1);
      }
      // Intialize the weight
      this->blobs_[0].reset(new Blob<Dtype>(1, 1, N_, K_));
      // fill the weights
      shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
								this->layer_param_.inner_product_param().weight_filler()));
      weight_filler->Fill(this->blobs_[0].get());
      // If necessary, intiialize and fill the bias term
      if (bias_term_) {
	this->blobs_[1].reset(new Blob<Dtype>(1, 1, 1, N_));
	shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
								this->layer_param_.inner_product_param().bias_filler()));
	bias_filler->Fill(this->blobs_[1].get());
      }
    }  // parameter initialization

    // Setting up the bias multiplier
    if (bias_term_) {
      bias_multiplier_.reset(new SyncedMemory(M_ * sizeof(Dtype)));
        Dtype* bias_multiplier_data =
	  reinterpret_cast<Dtype*>(bias_multiplier_->mutable_cpu_data());
        for (int i = 0; i < M_; ++i) {
	  bias_multiplier_data[i] = 1.;
        }
    }  
    //Setting up the eq_filter
    this->eq_filter_ = new Blob<Dtype>(bottom[0]->num(), 1, 1, bottom[0]->channels() * bottom[0]->height() * bottom[0]->width());
  }

  template <typename Dtype>
  Dtype InnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
					      vector<Blob<Dtype>*>* top) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = (*top)[0]->mutable_cpu_data();
    const Dtype* weight = this->blobs_[0]->cpu_data();
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
			  bottom_data, weight, (Dtype)0., top_data);
    if (bias_term_) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
			    reinterpret_cast<const Dtype*>(bias_multiplier_->cpu_data()),
			    this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
    }
    return Dtype(0);
  }

  template <typename Dtype>
  void InnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
					      const bool propagate_down,
					      vector<Blob<Dtype>*>* bottom) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = (*bottom)[0]->cpu_data();
    // Gradient with respect to weight
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
			  top_diff, bottom_data, (Dtype)0., this->blobs_[0]->mutable_cpu_diff());
    if (bias_term_) {
      // Gradient with respect to bias
      caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
			    reinterpret_cast<const Dtype*>(bias_multiplier_->cpu_data()), (Dtype)0.,
			    this->blobs_[1]->mutable_cpu_diff());
    }
    if (propagate_down) {
      // Gradient with respect to bottom data
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
			    top_diff, this->blobs_[0]->cpu_data(), (Dtype)0.,
			    (*bottom)[0]->mutable_cpu_diff());
    }
  }

  template <typename Dtype>
  void InnerProductLayer<Dtype>::UpdateEqFilter(const Blob<Dtype>* top_filter,
						const vector<Blob<Dtype>*>& input){
    //LOG(INFO)<<"Calculating Feedback Weights for "<<this->layer_param_.name();
    int top_output_num = top_filter->height();
    int outputsize = top_filter->width();
    int inputsize = input[0]->count() / input[0]->num();

    const Dtype* top_filter_data = (GPUMODE ? top_filter->gpu_data() : top_filter->cpu_data());

    const Dtype* weight_data = (GPUMODE ? this->blobs_[0]->gpu_data() : this->blobs_[0]->cpu_data());
    Dtype* eq_filter_data = (GPUMODE ? this->eq_filter_->mutable_gpu_data() : this->eq_filter_->mutable_cpu_data());

    //each input image may generate different filters
    for (int i = 0; i<this->eq_filter_->num(); i++){
      //Calculate the eq_filter_ by using matrix multiplication:
      //eq_filter_ = top_filter * weights (blobs_)
      if(GPUMODE){
	caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
			      top_output_num, inputsize, outputsize, (Dtype)1., top_filter_data, weight_data, (Dtype)0., eq_filter_data);
      }
      else{
	caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
			      top_output_num, inputsize, outputsize, (Dtype)1., top_filter_data, weight_data, (Dtype)0., eq_filter_data);
      }
      //next image
      top_filter_data += top_filter->offset(1);
      eq_filter_data += this->eq_filter_->offset(1);
    }
  }

  INSTANTIATE_CLASS(InnerProductLayer);

}  // namespace caffe
