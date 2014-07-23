// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

  template <typename Dtype>
  void ConvolutionLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
				      vector<Blob<Dtype>*>* top) {
    CHECK_EQ(bottom.size(), 1) << "Conv Layer takes a single blob as input.";
    CHECK_EQ(top->size(), 1) << "Conv Layer takes a single blob as output.";
    kernel_size_ = this->layer_param_.convolution_param().kernel_size();
    stride_ = this->layer_param_.convolution_param().stride();
    group_ = this->layer_param_.convolution_param().group();

    //for debug: what is the group?
    //LOG(INFO)<<"Group = "<<group_;
    //Mark: group_ is used to divide the input channels / filters into groups
    // For information, refer to Alex's cuda-convnet: LayerParameters#Convolution_layer
    pad_ = this->layer_param_.convolution_param().pad();
    num_ = bottom[0]->num();
    channels_ = bottom[0]->channels();
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    num_output_ = this->layer_param_.convolution_param().num_output();
    CHECK_GT(num_output_, 0);
    CHECK_EQ(channels_ % group_, 0);
    // The im2col result buffer would only hold one image at a time to avoid
    // overly large memory usage.
    int height_out = (height_ + 2 * pad_ - kernel_size_) / stride_ + 1;
    int width_out = (width_ + 2 * pad_ - kernel_size_) / stride_ + 1;
    col_buffer_.Reshape(
			1, channels_ * kernel_size_ * kernel_size_, height_out, width_out);
    // Set the parameters
    CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";
    bias_term_ = this->layer_param_.convolution_param().bias_term();
    // Figure out the dimensions for individual gemms.
    M_ = num_output_ / group_;
    K_ = channels_ * kernel_size_ * kernel_size_ / group_;
    N_ = height_out * width_out;
    (*top)[0]->Reshape(bottom[0]->num(), num_output_, height_out, width_out);
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
      this->blobs_[0].reset(new Blob<Dtype>(
					    num_output_, channels_ / group_, kernel_size_, kernel_size_));
      // fill the weights
      shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
								this->layer_param_.convolution_param().weight_filler()));
      weight_filler->Fill(this->blobs_[0].get());
      // If necessary, intiialize and fill the bias term
      if (bias_term_) {
	this->blobs_[1].reset(new Blob<Dtype>(1, 1, 1, num_output_));
	shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
								this->layer_param_.convolution_param().bias_filler()));
	bias_filler->Fill(this->blobs_[1].get());
      }
    }

    //for debug
    LOG(INFO)<<"Parameters: " << this->blobs_.size() << " * "<<this->blobs_[0]->num() << " * " << this->blobs_[0]->channels()
	     << " * " <<this->blobs_[0]->height() <<" * " <<this->blobs_[0]->width();

    // Set up the bias filler
    if (bias_term_) {
      bias_multiplier_.reset(new SyncedMemory(N_ * sizeof(Dtype)));
      Dtype* bias_multiplier_data =
        reinterpret_cast<Dtype*>(bias_multiplier_->mutable_cpu_data());
      for (int i = 0; i < N_; ++i) {
        bias_multiplier_data[i] = 1.;
      }
    }
  }


  template <typename Dtype>
  Dtype ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
					     vector<Blob<Dtype>*>* top) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = (*top)[0]->mutable_cpu_data();
    Dtype* col_data = col_buffer_.mutable_cpu_data();
    const Dtype* weight = this->blobs_[0]->cpu_data();

    /*
     * For information
     * M_ = num_output_ / group_
     * 	total number of output for a group
     * K_ = channels_ * kernel_size_ * kernel_size_ / group_;
     * 	The size of one filter
     * N_ = height_out * width_out;
     * 	The number of output for a single channel
     */
    int weight_offset = M_ * K_;		// The total size of the filters
    int col_offset = K_ * N_;			// the input needed to generate a channel of output
    int top_offset = M_ * N_;			// Total size of output
    for (int n = 0; n < num_; ++n) {
      // First, im2col
      // The column wised images is stored in col_data
      // size: 1, channels_ * kernel_size_ * kernel_size_, height_out, width_out
      im2col_cpu(bottom_data + bottom[0]->offset(n), channels_, height_,
		 width_, kernel_size_, pad_, stride_, col_data);
      // Second, innerproduct with groups
      for (int g = 0; g < group_; ++g) {
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
			      (Dtype)1., weight + weight_offset * g, col_data + col_offset * g,
			      (Dtype)0., top_data + (*top)[0]->offset(n) + top_offset * g);
      }
      // third, add bias
      if (bias_term_) {
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
			      N_, 1, (Dtype)1., this->blobs_[1]->cpu_data(),
			      reinterpret_cast<const Dtype*>(bias_multiplier_->cpu_data()),
			      (Dtype)1., top_data + (*top)[0]->offset(n));
      }
      bool test_flag = false;
      if(test_flag){
          //test functions including convolution, deconvolution
          Dtype* _input_data = bottom[0]->mutable_cpu_data() + bottom[0]->offset(n);
          int _output_num = this->num_output_;
          int _kernel_size = this->kernel_size_;
          int _pad = this->pad_;
          int _stride = this->stride_;
          Dtype* _filter = this->blobs_[0]->mutable_cpu_data();
          int _output_len = (*top)[0]->width() * (*top)[0]->height() * (*top)[0]->channels();
          Dtype* _output_data = new Dtype[_output_len];
          LOG(INFO)<<"[============ Testing convolution ===========]";
          convolution(_input_data, _filter, _output_data, _output_num, 
		      bottom[0]->channels(), bottom[0]->height(), bottom[0]->width(), _kernel_size, _pad, _stride);
          //calculate the error:
          int error_count = 0;
          for(int i = 0; i<_output_len; i++){
              Dtype _error = _output_data[i]-*((*top)[0]->mutable_cpu_data()+(*top)[0]->offset(n) + i);
              Dtype error = _error * _error;
              if (error > (Dtype) 0.01) error_count += 1;
          }
          LOG(INFO)<<"Error pixel ratio in testing convolution is: "<<error_count << " / "<<_output_len;

          //Test the deconvolution:
          LOG(INFO)<<"[=========== Testing deconvolution ==========]";
          int _input_len = bottom[0]->channels() * bottom[0]->width() * bottom[0]->height();
          Dtype* _deconv_output_data = new Dtype[_input_len];
          deconvolution(_output_data, _filter, _deconv_output_data,
              _output_num, bottom[0]->channels(), bottom[0]->height(), bottom[0]->width(),
              _kernel_size, _pad, _stride);
          //calculate error between _input_data and _deconv_output_data
          error_count = 0;
          for(int i = 0; i<_input_len; ++i) {
              Dtype _error = _deconv_output_data[i] - _input_data[i];
              Dtype error = _error * _error;
              if(error > 0.1) error_count += 1;
          }
          LOG(INFO)<<"Error pixel ratio in testing deconvolution is: "<<error_count << " / "<<_input_len;
	  delete [] _output_data;
	  delete [] _deconv_output_data;      
      }
    }
    return Dtype(0.);
  }

  template <typename Dtype>
  void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
					     const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* weight = this->blobs_[0]->cpu_data();
    Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
    const Dtype* bottom_data = (*bottom)[0]->cpu_data();
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    Dtype* col_data = col_buffer_.mutable_cpu_data();
    Dtype* col_diff = col_buffer_.mutable_cpu_diff();
    // bias gradient if necessary
    Dtype* bias_diff = NULL;

    if (bias_term_) {
      bias_diff = this->blobs_[1]->mutable_cpu_diff();
      memset(bias_diff, 0, sizeof(Dtype) * this->blobs_[1]->count());
      for (int n = 0; n < num_; ++n) {
	caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, N_,
			      1., top_diff + top[0]->offset(n),
			      reinterpret_cast<const Dtype*>(bias_multiplier_->cpu_data()), 1.,
			      bias_diff);
      }
    }

    int weight_offset = M_ * K_;
    int col_offset = K_ * N_;
    int top_offset = M_ * N_;
    memset(weight_diff, 0, sizeof(Dtype) * this->blobs_[0]->count());
    for (int n = 0; n < num_; ++n) {
      // since we saved memory in the forward pass by not storing all col data,
      // we will need to recompute them.
      im2col_cpu(bottom_data + (*bottom)[0]->offset(n), channels_, height_,
		 width_, kernel_size_, pad_, stride_, col_data);
      // gradient w.r.t. weight. Note that we will accumulate diffs.
      for (int g = 0; g < group_; ++g) {
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
			      (Dtype)1., top_diff + top[0]->offset(n) + top_offset * g,
			      col_data + col_offset * g, (Dtype)1.,
			      weight_diff + weight_offset * g);
      }
      // gradient w.r.t. bottom data, if necessary
      if (propagate_down) {
	for (int g = 0; g < group_; ++g) {
	  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
				(Dtype)1., weight + weight_offset * g,
				top_diff + top[0]->offset(n) + top_offset * g,
				(Dtype)0., col_diff + col_offset * g);
	}
	// col2im back to the data
	col2im_cpu(col_diff, channels_, height_, width_, kernel_size_, pad_,
		   stride_, bottom_diff + (*bottom)[0]->offset(n));
      }
    }
  }

  //Checked, no bugs
  template <typename Dtype>
  void ConvolutionLayer<Dtype>::UpdateEqFilter(const Blob<Dtype>* top_filter,
      const vector<Blob<Dtype>*>& input) {
    int input_size = channels_ * width_ * height_;
    int output_size = top_filter->height();
    //the final output only has one channel    
    int output_channel = top_filter->channels();

    this->eq_filter_ = new Blob<Dtype>(input[0]->num(), output_channel, output_size, input_size);
    Dtype* eq_filter_data = this->eq_filter_->mutable_cpu_data();
    const Dtype* top_filter_data = top_filter->cpu_data();
    int height_out = (height_ + 2 * pad_ - kernel_size_) / stride_ + 1;
    int width_out = (width_ + 2 * pad_ - kernel_size_) / stride_ + 1;

    //for debug
    LOG(INFO)<<"[Size of eq_filter: input/output] "<<height_<<"*"<<width_<<"/"
	     <<height_out <<"*"<<width_out;

    for (int n = 0; n<input[0]->num(); n++){
      for (int o = 0; o<output_size; ++o){
	deconvolution(top_filter_data + top_filter->offset(n) + o * top_filter->width(),           //top_filter
		      this->blobs_[0]->mutable_cpu_data(),                                         //convolution filters
		      eq_filter_data + this->eq_filter_->offset(n) + o * input_size,               //eq_filter
		      num_output_, height_out, width_out,
		      channels_, kernel_size_, pad_, stride_);
	/****************************************************************************/
	//Test for deconvolution
	Dtype error = 0;
	Dtype* input_data_ptr = input[0]->mutable_cpu_data() + input[0]->offset(n);
	//calculate convolution:
	Dtype* conv_rslt = new Dtype[top_filter->width()];
	convolution(input_data_ptr, this->blobs_[0]->mutable_cpu_data(), conv_rslt, num_output_, 
		   channels_, height_, width_, 
		   kernel_size_, pad_, stride_);
	Dtype conv_score = caffe_cpu_dot<Dtype>(top_filter->width(), conv_rslt, 
						top_filter_data + top_filter->offset(n) + o* top_filter->width());
	//calculate using deconvolution
	Dtype deconv_score = caffe_cpu_dot<Dtype>(input_size, input_data_ptr, 
						  eq_filter_data + this->eq_filter_->offset(n) + o * input_size);
	error = (conv_score - deconv_score) * (conv_score - deconv_score);
	LOG(INFO)<<"Error of eq_filter on Image "<<n<<" at output "<<o<<" is "<<error<<"("<<conv_score<<" / "<<deconv_score<<")";
	delete [] conv_rslt;
      } //output num
    } // for each input image
  }


  template <typename Dtype>
  void ConvolutionLayer<Dtype>::filter_transpose(Dtype* filter,
						 int output_num, int channels, int kernel_size,
						 Dtype* t_filter) {
    int _filter_size = kernel_size * kernel_size;
    Dtype* _r_filter = new Dtype[_filter_size * channels * output_num];
    //Reverse the filter: mirror in both horizon and vertical
    for(int c = 0; c < channels * output_num; ++c) {
        for(int i = 0; i<_filter_size; ++i) {
            *(_r_filter + c * _filter_size + i) = *(filter + c * _filter_size + _filter_size - i - 1);
        }
    }
    for (int o = 0; o<output_num; ++o){
      for (int c = 0; c<channels; ++c){
	//memcpy: des, src, num = sizeof(Dtype) * kernel_size * kernel_sizel
	int filter_offset = (o * channels + c ) * _filter_size ;
	int t_filter_offset = (c * output_num + o) * _filter_size;
	memcpy(t_filter + t_filter_offset, _r_filter + filter_offset, sizeof(Dtype) * _filter_size);
      } // for each channel
    } // for each output
    delete [] _r_filter;
  }

  //Checked: no bugs
  template <typename Dtype>
  void ConvolutionLayer<Dtype>::convolution(Dtype* input, Dtype* filter,
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
     * 	total number of output for a group
     * K_ = channels_ * kernel_size_ * kernel_size_ / group_;
     * 	The size of one filter
     * N_ = height_out * width_out;
     * 	The number of output for a single channel
     */
    Dtype* col_data = new Dtype[channels * kernel_size * kernel_size * height_out * width_out];
    //im2col
    im2col_cpu(input, channels, height,
	       width, kernel_size, pad, stride, col_data);
    //Performing inner product
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
			  (Dtype)1., filter, col_data,
			  (Dtype)0., output);

    delete [] col_data;
  }

  //Function to perform deconvolution:
  //response: the response after convolution, a 3-dimensional matrix (in shape of Blob)
  //	- it is a row in the top-filter
  //  - the size of response is output_num * output_height * output_width
  //filter: convolution filter, a 4-dimensional matrix (in shape of Blob)
  //output: the deconvolution results, of the same size of input image, a 3-dimensional matrix
  //	- channels is the number of channels in the input image
  template <typename Dtype>
  void ConvolutionLayer<Dtype>::deconvolution(const Dtype* response, Dtype* filter, Dtype* output,
					      int output_num, int output_height, int output_width,
					      int channels, int kernel_size, int pad, int stride) {
    //1. striding and padding
    int s_width = output_width * stride;
    int s_height = output_height * stride;
    Dtype* s_response = new Dtype[s_width * s_height * output_num];
    //1.1. initialize all 0
    memset(s_response, 0, sizeof(Dtype) * s_width * s_height * output_num);
    //1.2. expanding
    for (int c = 0; c<output_num; ++c){
      for(int h = 0; h<output_height; ++h) {
	for(int w = 0; w<output_width; ++w){
	  int s_offset = c * s_width * s_height                       	//layer
	    + h * stride * s_width  					//stride rows
	    + w * stride;						//stride offset
	  *(s_response + s_offset) = *response;
	  response += 1;
	}// each column
      }// each row
    }// for each output channel
    //2. filter_transponse
    Dtype* t_filter = new Dtype[kernel_size * kernel_size * channels * output_num];
    filter_transpose(filter, output_num, channels, kernel_size, t_filter);
    //3. convolution: the input channels is the output_num in deconvolution, and vice versa.
    int deconv_pad = kernel_size - 1;
    if (pad > 0){
      int _height = (s_height + 2*deconv_pad - kernel_size + 1);
      int original_height = this->height_;
      int original_width = this->width_;
      int _width = (s_width + 2*deconv_pad - kernel_size + 1);
      Dtype* deconv_output = new Dtype[channels * _height * _width];
      convolution(s_response, t_filter, deconv_output, channels, output_num, s_height, s_width, kernel_size, deconv_pad, 1);
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
      delete [] deconv_output;
    }
    else{
      convolution(s_response, t_filter, output, channels, output_num, s_height, s_width, kernel_size, deconv_pad, 1);
    }
    //5. Cleaning:
    delete [] t_filter;
    delete [] s_response;
  }

  INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
