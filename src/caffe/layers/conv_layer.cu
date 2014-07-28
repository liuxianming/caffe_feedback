// Copyright 2014 BVLC and contributors.
#include <cuda_runtime.h>
#include <vector>

#include "caffe/syncedmem.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

  template <typename Dtype>
  Dtype ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = (*top)[0]->mutable_gpu_data();
    Dtype* col_data = col_buffer_.mutable_gpu_data();
    const Dtype* weight = this->blobs_[0]->gpu_data();
    int weight_offset = M_ * K_;
    int col_offset = K_ * N_;
    int top_offset = M_ * N_;
    for (int n = 0; n < num_; ++n) {
        // First, im2col
        im2col_gpu(bottom_data + bottom[0]->offset(n), channels_, height_,
            width_, kernel_size_, pad_, stride_, col_data);
        // Second, innerproduct with groups
        for (int g = 0; g < group_; ++g) {
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
                (Dtype)1., weight + weight_offset * g, col_data + col_offset * g,
                (Dtype)0., top_data + (*top)[0]->offset(n) + top_offset * g);
        }
        // third, add bias
        if (bias_term_) {
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
                N_, 1, (Dtype)1., this->blobs_[1]->gpu_data(),
                reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
                (Dtype)1., top_data + (*top)[0]->offset(n));
        }
    }
    return Dtype(0.);
  }

  template <typename Dtype>
  void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* weight = this->blobs_[0]->gpu_data();
    Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
    const Dtype* bottom_data = (*bottom)[0]->gpu_data();
    Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
    Dtype* col_data = col_buffer_.mutable_gpu_data();
    Dtype* col_diff = col_buffer_.mutable_gpu_diff();
    // bias gradient if necessary
    Dtype* bias_diff = NULL;

    if (bias_term_) {
        bias_diff = this->blobs_[1]->mutable_gpu_diff();
        CUDA_CHECK(cudaMemset(bias_diff, 0,
            sizeof(Dtype) * this->blobs_[1]->count()));
        for (int n = 0; n < num_; ++n) {
            caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, N_,
                1., top_diff + top[0]->offset(n),
                reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
                1., bias_diff);
        }
    }

    int weight_offset = M_ * K_;
    int col_offset = K_ * N_;
    int top_offset = M_ * N_;
    CUDA_CHECK(cudaMemset(weight_diff, 0,
        sizeof(Dtype) * this->blobs_[0]->count()));
    for (int n = 0; n < num_; ++n) {
        // since we saved memory in the forward pass by not storing all col data,
        // we will need to recompute them.
        im2col_gpu(bottom_data + (*bottom)[0]->offset(n), channels_, height_,
            width_, kernel_size_, pad_, stride_, col_data);
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        for (int g = 0; g < group_; ++g) {
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
                (Dtype)1., top_diff + top[0]->offset(n) + top_offset * g,
                col_data + col_offset * g, (Dtype)1.,
                weight_diff + weight_offset * g);
        }
        // gradient w.r.t. bottom data, if necessary
        if (propagate_down) {
            for (int g = 0; g < group_; ++g) {
                caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
                    (Dtype)1., weight + weight_offset * g,
                    top_diff + top[0]->offset(n) + top_offset * g,
                    (Dtype)0., col_diff + col_offset * g);
            }
            // col2im back to the data
            col2im_gpu(col_diff, channels_, height_, width_, kernel_size_, pad_,
                stride_, bottom_diff + (*bottom)[0]->offset(n));
        }
    }
  }

  template <typename Dtype>
  void ConvolutionLayer<Dtype>::convolution_gpu(Dtype* input, Dtype* filter,
      Dtype* output,
      int output_num, int channels, int height, int width,
      int kernel_size, int pad, int stride) {
    int height_out = (height + 2 * pad - kernel_size) / stride + 1;
    int width_out = (width + 2 * pad - kernel_size) / stride + 1;
    int M_ = output_num;
    int K_ = channels * kernel_size * kernel_size;
    int N_ = height_out * width_out;
    /*
     * For information
     * M_ = num_output_ / group_
     *  total number of output for a group
     * K_ = channels_ * kernel_size_ * kernel_size_ / group_;
     *  The size of one filter
     * N_ = height_out * width_out;
     *  The number of output for a single channel
     */
    shared_ptr<SyncedMemory> col_data_;
    col_data_.reset(new SyncedMemory(channels * kernel_size * kernel_size * height_out * width_out * sizeof(Dtype)));
    Dtype* col_data = reinterpret_cast<Dtype*>(col_data_->mutable_gpu_data());
    //im2col
    im2col_gpu(input, channels, height,
        width, kernel_size, pad, stride, col_data);
    //Performing inner product
    ///*
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
        (Dtype)1., filter, col_data,
        (Dtype)0., output);
    //delete [] col_data;
  }

  template <typename Dtype>
  void ConvolutionLayer<Dtype>::deconvolution_gpu(const Dtype* response, Dtype* filter, Dtype* output,
      int output_num, int output_height, int output_width,
      int channels, int kernel_size, int pad, int stride) {
    //1. striding and padding
    int s_width = output_width * stride;
    int s_height = output_height * stride;
    shared_ptr<SyncedMemory> s_response_;
    s_response_.reset(new SyncedMemory(s_width * s_height * output_num * sizeof(Dtype)));
    Dtype* s_response = reinterpret_cast<Dtype*>(s_response_->mutable_cpu_data());
    //1.1. initialize all 0
    memset(s_response_->mutable_cpu_data(), 0, sizeof(Dtype) * s_width * s_height * output_num);
    //1.2. expanding
    for (int c = 0; c<output_num; ++c){
        for(int h = 0; h<output_height; ++h) {
            for(int w = 0; w<output_width; ++w){
                int s_offset = c * s_width * s_height                           //layer
                    + h * stride * s_width                                      //stride rows
                    + w * stride;                                               //stride offset
                *(s_response + s_offset) = *response;
                response += 1;
            }// each column
        }// each row
    }// for each output channel
    //2. filter_transponse
    shared_ptr<SyncedMemory> t_filter_;
    t_filter_.reset(new SyncedMemory(kernel_size * kernel_size * channels * output_num * sizeof(Dtype)));
    filter_transpose(filter, output_num, channels, kernel_size, reinterpret_cast<Dtype*>(t_filter_->mutable_cpu_data()));
    //3. convolution: the input channels is the output_num in deconvolution, and vice versa.
    int deconv_pad = kernel_size - 1;

    int _height = (s_height + 2*deconv_pad - kernel_size + 1);
    int original_height = this->height_;
    int original_width = this->width_;
    int _width = (s_width + 2*deconv_pad - kernel_size + 1);
    shared_ptr<SyncedMemory> deconv_output_;
    deconv_output_.reset(new SyncedMemory(channels * _height * _width * sizeof(Dtype)));

    convolution_gpu(reinterpret_cast<Dtype*>(s_response_->mutable_gpu_data()), reinterpret_cast<Dtype*>(t_filter_->mutable_gpu_data()),
        reinterpret_cast<Dtype*>(deconv_output_->mutable_gpu_data()), channels, output_num, s_height, s_width, kernel_size, deconv_pad, 1);

    Dtype* deconv_output = reinterpret_cast<Dtype*>(deconv_output_->mutable_cpu_data());
    //4. un-padding
    for (int c = 0; c<channels; ++c) {
        for (int h = 0; h<original_height; ++h){
            int offset = c * _height * _width
                + _width * (h + pad)
                + pad;
            int original_offset = c * original_height * original_width
                + h * original_width;
            memcpy(output + original_offset, deconv_output + offset, sizeof(Dtype) * original_width);
        }
    }
    // For shared_ptr, there is no need to clean the memory
  }


  INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
