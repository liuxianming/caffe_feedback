// Copyright Xianming Liu, July 11 2014.

#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>
// #include <cblas.h>
#include <algorithm>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/feedback_net.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/insert_splits.hpp"
#include "caffe/util/upgrade_proto.hpp"
#include "caffe/util/math_functions.hpp"

using std::pair;
using std::map;
using std::set;
using std::string;

namespace caffe{

  template<typename Dtype>
  void FeedbackNet<Dtype>::InitVisualization(){
  }

  template<typename Dtype>
  Blob<Dtype>* FeedbackNet<Dtype>::VisualizeSingleNeuron(int startLayerIdx,
							 int startChannelIdx, 
							 int startOffset, 
							 bool weight_flag) {
    UpdateEqFilter(startLayerIdx, startChannelIdx, startOffset);
    //Get the eq_filter_ at the data layer as the output;
    Blob<Dtype>* eq_filter_output  = this->eq_filter_top_.back();
    //Re-organize the eq_filter_output to get this->visualization_
    Blob<Dtype>* input_blob = (this->blobs_[0]).get();
    Blob<Dtype>* _visualization 
      = new Blob<Dtype>(input_blob->num(), input_blob->channels(),
			input_blob->height(), input_blob->width());
    memcpy(_visualization->mutable_cpu_data(), eq_filter_output->cpu_data(), 
	   sizeof(Dtype)*eq_filter_output->count());
    //Find weight
    Blob<Dtype>* _filter_output = (this->top_vecs_[startLayerIdx_])[0];
    Dtype* output_weights = new Dtype[_filter_output->num()];
    for(int i = 0; i<_filter_output->num(); ++i) {
      output_weights[i] = *(_filter_output->mutable_cpu_data() 
			    + _filter_output->offset(i, startChannelIdx) + startOffset);
      //Test: to output the predicted values and the feedforward values
      Dtype predict_value = caffe_cpu_dot(input_blob->count() / input_blob->num(),
					  input_blob->cpu_data() + input_blob->offset(i),
					  _visualization->cpu_data() 
					  + _visualization->offset(i));
      LOG(INFO)<<"The difference between predicted output and the feedforward output is: "
	       <<predict_value<<" : "<<output_weights[i];
    }
    if(weight_flag) {
      _visualization->multiply( output_weights);
    }
    return _visualization;
  }

  template<typename Dtype>
  void FeedbackNet<Dtype>::UpdateEqFilter(int startLayerIdx, 
					  int startChannelIdx, 
					  int startOffset){
    this->startLayerIdx_ = startLayerIdx;
    this->startChannelIdx_ = startChannelIdx;
    this->startOffset_ = startOffset;

    //Generate top mask
    generateStartTopFilter(startOffset);
    UpdateEqFilter();
  }

  template<typename Dtype>
  void FeedbackNet<Dtype>::Visualize(int startLayerIdx, int startChannelIdx, 
				     int heightOffset, int widthOffset, bool test_flag){
    //The visualization must be completed by performing Forward() in advance
    if (this->forwardCompleteFlag == false) {
      LOG(ERROR)<<"[Error] Have to input image into the network to complete visualization";
      return;
    }
    if(startLayerIdx < 0 ){
      LOG(ERROR)<<"Wrong Layer for visualization: "<<startLayerIdx <<"-> Data Layer" ;
      return;
    }
    if (startLayerIdx == 0){
      //By default, start from the last layer (fc8 or softmax layer)
      startLayerIdx = this->layers_.size() - 1;
    }

    this->startLayerIdx_ = startLayerIdx;
    this->startChannelIdx_ = startChannelIdx;
    
    //0. Construct the layer index array for visualization
    InitVisualization();
    int output_layer_width = (this->top_vecs_[startLayerIdx_])[0]->width();
    int output_layer_height = (this->top_vecs_[startLayerIdx_])[0]->height();
    int output_layer_channel = (this->top_vecs_[startLayerIdx_])[0]->channels();
    LOG(INFO)<<output_layer_width<<":"<<output_layer_height<<":"<<output_layer_channel;
    if(heightOffset >= 0 && widthOffset >= 0) {

      this->startOffset_ = widthOffset + heightOffset * output_layer_width;
      if (this->startOffset_ >= (this->top_vecs_[startLayerIdx])[0]->width() * (this->top_vecs_[startLayerIdx_])[0]->height()) {
	LOG(ERROR)<<"Wrong output neuron!";
	return;
      }
      this->visualization_ = VisualizeSingleNeuron(startLayerIdx_, startChannelIdx_, this->startOffset_, false);
      if(test_flag){
	Dtype error = test_eq_filter(1, this->eq_filter_top_.back());
	LOG(INFO)<<"[ERROR for neuron #"<<this->startOffset_<<"] = "<<error;
      }
    }
    else if(heightOffset >= 0 && widthOffset < 0){
      //visualize single row
      this->visualization_ = VisualizeSingleRowNeurons(startLayerIdx, startChannelIdx, heightOffset);
    }
    else if(heightOffset < 0 && widthOffset >= 0){
      //visualize single col
      this->visualization_ = VisualizeSingleColNeurons(startLayerIdx, startChannelIdx, widthOffset);
    }
    else {
      Blob<Dtype>* input_blob = (this->blobs_[0]).get();
      Blob<Dtype>* _visualization = new Blob<Dtype>(input_blob->num(), input_blob->channels(),
						    input_blob->height(), input_blob->width());
      memset(_visualization->mutable_cpu_data(), 0, sizeof(Dtype)*_visualization->count());

      for(int h=0; h<output_layer_height; ++h) {
	_visualization->add( *(VisualizeSingleRowNeurons(startLayerIdx, startChannelIdx, h)));
      }
      this->visualization_ = _visualization;
    }
  }

  template<typename Dtype>
  void FeedbackNet<Dtype>::Visualize(string startLayer, int startChannelIdx, 
				     int heightOffset, int widthOffset, 
				     bool test_flag){
    int startLayerIdx = this->layer_names_index_[startLayer];

    this->Visualize(startLayerIdx, startChannelIdx, heightOffset, widthOffset,  test_flag);
  }

  template<typename Dtype>
  void FeedbackNet<Dtype>::SearchTopKNeurons(int startLayerIdx, int k, 
					     int* channel_offset, int* in_channel_offset){
    Blob<Dtype>* output_blobs = this->top_vecs_[startLayerIdx][0];
    int output_blob_channel_size = output_blobs->width() * output_blobs->height();
    Dtype* output_blobs_data = output_blobs->mutable_cpu_data();
    //sort the output for the first image in mini-batch
    vector<Dtype> sorted_vector(output_blobs_data, output_blobs_data 
				+ output_blobs->count()/output_blobs->num());
    std::sort(sorted_vector.begin(), sorted_vector.end());
    //get the top k
    typename vector<Dtype>::iterator pos = sorted_vector.end();
    for(int i = 0; i<k; ++i){
      pos --; 
      Dtype _val = *pos;
      Dtype* p = std::find(output_blobs_data, output_blobs_data + output_blobs->count(), _val);
      int offset = p - output_blobs_data;
      channel_offset[i] = offset / output_blob_channel_size;
      in_channel_offset[i] = offset % output_blob_channel_size;
    }
  }

  template<typename Dtype>
  void FeedbackNet<Dtype>::VisualizeTopKNeurons(string startLayer, int k, bool weight_flag){
    int startLayerIdx = this->layer_names_index_[startLayer];
    this->VisualizeTopKNeurons(startLayerIdx, k);
  }

  template<typename Dtype>
  void FeedbackNet<Dtype>::VisualizeTopKNeurons(int startLayerIdx, int k, bool weight_flag){
    this->startLayerIdx_ = startLayerIdx;
    if(this->blobs_[0]->num() > 1){
      LOG(ERROR)<<"This function only supports the input batch with size 1";
      return;
    }
    Blob<Dtype>* input_blob = (this->blobs_[0]).get();
    Blob<Dtype>* _visualization 
      = new Blob<Dtype>(input_blob->num(), input_blob->channels(),
			input_blob->height(), input_blob->width());
    memset(_visualization->mutable_cpu_data(), 0, sizeof(Dtype)*_visualization->count());

    int* channel_offset = new int[k];
    int* in_channel_offset = new int[k];
    SearchTopKNeurons(startLayerIdx, k, channel_offset, in_channel_offset);
    for(int i = 0; i<k; ++i){
      LOG(INFO)<<"Visualize using "<< channel_offset[i] <<" / "<<in_channel_offset[i];
      this->startChannelIdx_ = channel_offset[i];
      this->startOffset_ = in_channel_offset[i];
      _visualization ->add(*(VisualizeSingleNeuron(startLayerIdx, 
						   this->startChannelIdx_, 
						   this->startOffset_, 
						   weight_flag)));
    }
    this->visualization_ = _visualization;
  }

  template<typename Dtype>
  Blob<Dtype>* FeedbackNet<Dtype>::VisualizeSingleRowNeurons(int startLayerIdx, int startChannelIdx, int heightOffset) {
    int output_layer_width = (this->top_vecs_[startLayerIdx])[0]->width();
    int output_layer_height = (this->top_vecs_[startLayerIdx])[0]->height();

    Blob<Dtype>* input_blob = (this->blobs_[0]).get();
    Blob<Dtype>* _visualization 
      = new Blob<Dtype>(input_blob->num(), input_blob->channels(),
			input_blob->height(), input_blob->width());
    memset(_visualization->mutable_cpu_data(), 0, sizeof(Dtype)*_visualization->count());

    if(heightOffset < 0 || heightOffset >= output_layer_height) {
      LOG(ERROR)<<"Wrong Height offset";
    }
    else{
        for (int i = 0; i<output_layer_width; ++i) {
            int startOffset = i + output_layer_width * heightOffset;
            _visualization->add(*(VisualizeSingleNeuron(startLayerIdx, startChannelIdx, 
							startOffset, true)));
        }
    }
    return _visualization;
  }

  template<typename Dtype>
  Blob<Dtype>* FeedbackNet<Dtype>::VisualizeSingleColNeurons(int startLayerIdx, int startChannelIdx, int widthOffset) {
    int output_layer_width = (this->top_vecs_[startLayerIdx])[0]->width();
    int output_layer_height = (this->top_vecs_[startLayerIdx])[0]->height();

    Blob<Dtype>* input_blob = (this->blobs_[0]).get();
    Blob<Dtype>* _visualization 
      = new Blob<Dtype>(input_blob->num(), input_blob->channels(),
			input_blob->height(), input_blob->width());
    memset(_visualization->mutable_cpu_data(), 0, sizeof(Dtype)*_visualization->count());

    if(widthOffset < 0 || widthOffset >= output_layer_width) {
        LOG(ERROR)<<"Wrong Height offset";
    }
    else{
        for (int i = 0; i<output_layer_height; ++i) {
            int startOffset = i + output_layer_width + widthOffset;;
            _visualization->add(*(VisualizeSingleNeuron(startLayerIdx, startChannelIdx, 
							startOffset, true)));
        }
    }
    return _visualization;
}

  template<typename Dtype>
  void FeedbackNet<Dtype>::generateStartTopFilter(int startOffset){
    //Generate the top_filter_ for the start layer of visualization
    //The output of the top_filter is only one neuron,
    Blob<Dtype>* _top_output_blob = (this->top_vecs_[startLayerIdx_])[0];
    long _input_num 
      = _top_output_blob->channels() 
      * _top_output_blob->height()
      * _top_output_blob->width();
    long channel_offset = _top_output_blob->height() * _top_output_blob->width();
    long _img_num = this->blobs_[0]->num();

    start_top_filter_ = new Blob<Dtype>(_img_num, 1, 1, _input_num);

    //Initialize the value of start_top_filter_
    Dtype* start_top_filter_data = start_top_filter_->mutable_cpu_data();

    memset(start_top_filter_data, 0, sizeof(Dtype) * start_top_filter_->count());

    for (int n = 0; n<start_top_filter_->num(); n++) {
      *(start_top_filter_data + channel_offset * this->startChannelIdx_ + startOffset) 
	= Dtype(1.);
      start_top_filter_data += start_top_filter_->offset(1, 0);
    }
  }

  template<typename Dtype>
  void FeedbackNet<Dtype>::UpdateEqFilter(){
    //firs build top_filter vector
    this->eq_filter_top_.clear();
    this->eq_filter_top_.push_back(start_top_filter_);
    for(int i = this->startLayerIdx_; i > 0; --i){
      //LOG(INFO)<<"Processing Layer ["<<this->layers_[i]->layer_param().name()<<"]";
      string blob_name = this->blob_names_[(this->bottom_id_vecs_[i])[0]];
      //LOG(INFO)<<"Using BLOB ["<<blob_name<<"] as input";
      this->layers_[i]->UpdateEqFilter(this->eq_filter_top_.back(), this->bottom_vecs_[i]);
      this->eq_filter_top_.push_back(this->layers_[i]->eq_filter());

      //test
      bool test_flag = false;
      if (test_flag) {
	Dtype error = test_eq_filter(i, this->eq_filter_top_.back());
	LOG(INFO)<<"Error of Layer "<<this->layer_names_[i]<<" is "
		 <<error<<" : "<<((error < (Dtype) 16.) ? "OK" : "FAIL");
      }
    }
  }

  template<typename Dtype>
  void FeedbackNet<Dtype>::DrawVisualization(string dir, string prefix) {
    std::ostringstream convert;
    //Normalization:
    for(int n = 0; n<this->visualization_->num(); ++n) {
      for (int c = 0; c<this->visualization_->channels(); c++) {
	ImageNormalization<Dtype>(this->visualization_->mutable_cpu_data() 
				  + this->visualization_->offset(n, c),
				  this->visualization_->offset(0,1), 
				  (Dtype)100, (Dtype)20.0);
      }
    }
    for (int n = 0; n<visualization_->num(); n++) {
      convert<<dir<<prefix<<n;
      string filename = convert.str();
      if (visualization_->channels() == 3 || visualization_->channels() == 1){
	//Visualize as RGB / Grey image
	filename = filename + ".jpg";
	const int _height = visualization_->height();
	const int _width = visualization_->width();
	const int _channel = visualization_->channels();
	WriteDataToImage<Dtype>(filename, _channel, _height, _width,
			 visualization_->mutable_cpu_data() + visualization_->offset(n) );
      }
      else{
	//Visualize each channel as a separate image
	for(int c = 0; c<visualization_->channels(); c++) {
	  convert<<filename<<"_"<<c<<".jpg";
	  string filename = convert.str();
	  const int _height = visualization_->height();
	  const int _width = visualization_->width();
	  const int _channel = visualization_->channels();
	  WriteDataToImage<Dtype>(filename, _channel, _height, _width,
				  visualization_->mutable_cpu_data() 
				  + visualization_->offset(n, c) );
	}
      }
    }
  }

  template<typename Dtype>
  Dtype FeedbackNet<Dtype>::test_eq_filter(int _layer_idx, Blob<Dtype>* eq_filter) {
    Blob<Dtype>* feedforward_output_blob = (this->top_vecs_[this->startLayerIdx_])[0];

    Blob<Dtype>* input_blob = (this->bottom_vecs_[_layer_idx])[0];
    LOG(INFO)<<"Testing using blob "
	     <<this->blob_names_[this->bottom_id_vecs_[_layer_idx][0]] <<" as input";
    LOG(INFO)<<"Size: "
	     <<input_blob->channels() 
	     << " * " << input_blob->height() 
	     << " * "<<input_blob->width();
    Dtype error = (Dtype) 0.;

    for(int n = 0; n<input_blob->num(); ++n) {
      Dtype output_val = *(feedforward_output_blob->mutable_cpu_data()
			   + feedforward_output_blob->offset(n, this->startChannelIdx_)
			   + this->startOffset_);
      Dtype* input_data = input_blob->mutable_cpu_data() + input_blob->offset(n);
      Dtype* eq_filter_data = eq_filter->mutable_cpu_data() + eq_filter->offset(n);
      int len = input_blob->channels() * input_blob->height() * input_blob->width();
      Dtype predicted_val = caffe_cpu_dot<Dtype>(len, input_data, eq_filter_data);

      error += (output_val - predicted_val) * (output_val - predicted_val);
      //LOG(INFO)<<"OUTTEST :"<<output_val << " / "<<predicted_val;
    }
    return error;
  }

  template<typename Dtype>
  const vector<Blob<Dtype>*>& FeedbackNet<Dtype>::FeedbackForwardPrefilled(string startLayer, int channel, int offset) {
    int startLayerIdx = this->layer_names_index_[startLayer];
    FeedbackForwardPrefilled(NULL, startLayerIdx, channel, offset);
  }

  template<typename Dtype>
  const vector<Blob<Dtype>*>& FeedbackNet<Dtype>::FeedbackForwardPrefilled(Dtype* loss, int startLayerIdx, int channel, int offset){
    this->forwardCompleteFlag = true;
    int k = 5;
    int* k_channel = new int[k];
    int* k_offset = new int[k];
    //reset layers
    for (int i = 0; i < this->layers_.size(); ++i) {
      this->layers_[i]->Reset();
    }

    if (loss != NULL) {
      *loss = Dtype(0.);
    }
    for (int i = 0; i < this->layers_.size(); ++i) {
      Dtype layer_loss = this->layers_[i]->Forward(this->bottom_vecs_[i], &(this->top_vecs_[i]));
      if (loss != NULL) {
	*loss += layer_loss;
      }
    }

    if(startLayerIdx == -1){
      startLayerIdx = this->layers_.size()-1;
    }
    SearchTopKNeurons(startLayerIdx, 5, k_channel, k_offset);
    if(channel == -1 && offset == -1) {
      //get the largest output category
      channel = k_channel[0];
      offset = k_offset[0];
    }
    LOG(INFO)<<"Using the neuron at "<<channel << " / "<<offset<<" As target";
    //pick the value of output
    Dtype output_sum = (Dtype) 0.;
    Blob<Dtype>* top_output_blob = (this->top_vecs_[startLayerIdx])[0];
    Dtype* output_value = new Dtype[top_output_blob->num()];
    LOG(INFO)<<"Initial Output Values:";
    for(int n = 0; n<top_output_blob->num(); ++n){
      output_value[n] = *(top_output_blob->cpu_data() 
			  + top_output_blob->offset(n, channel) + offset);
      output_sum += output_value[n];
      LOG(INFO)<<output_value[n];
      std::ostringstream convert;
      for(int i = 0; i<k; i++){
          Dtype value = *(top_output_blob->cpu_data()
              + top_output_blob->offset(n, k_channel[i]) + k_offset[i]);
          convert <<"[Channel: " <<k_channel[i] <<":"<< value << "] ";
      }
      string values = convert.str();
      LOG(INFO)<<values;
    }
    int iteration  = 0;
    while(true){
      //UpdateEqFilter
      UpdateEqFilter(startLayerIdx, channel, offset);
      //Forward - don't deal with data layer
      for (int i = 1; i < this->layers_.size(); ++i) {
	Dtype layer_loss 
	  = this->layers_[i]->Forward(this->bottom_vecs_[i], &(this->top_vecs_[i]));
      }
      SearchTopKNeurons(startLayerIdx, k, k_channel, k_offset);
      LOG(INFO)<<"Feedback Iteration"<<iteration;
      Dtype new_output_sum = (Dtype) 0.;
      for(int n = 0; n<top_output_blob->num(); ++n){
	output_value[n] = *(top_output_blob->cpu_data() 
			    + top_output_blob->offset(n, channel) + offset);
	new_output_sum += output_value[n];
	LOG(INFO)<<output_value[n];
	std::ostringstream convert;
	for(int i = 0; i<k; i++){
	    Dtype value = *(top_output_blob->cpu_data()
	        + top_output_blob->offset(n, k_channel[i]) + k_offset[i]);
	    convert << k_channel[i] <<":"<< value << " ";
	}
	string values = convert.str();
	LOG(INFO)<<values;
      }
      iteration ++;
      Dtype converge = (output_sum - new_output_sum) * (output_sum - new_output_sum);
      if(converge <= 1 || iteration > 5){
	//Doesnot allow too many iterations
	LOG(INFO)<<"Converge in " <<iteration << " iterations!";
	break;
      }
      else{
	output_sum = new_output_sum;
      }
    }
    return this->net_output_blobs_;
  }

  INSTANTIATE_CLASS(FeedbackNet);

} //namespace caffe
