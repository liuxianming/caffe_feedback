// Copyright Xianming Liu, July 11 2014.

#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

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
    //setup the eq_filter_ flow 
    /*
     * Follow the similar work flow as in Net.Init(),
     * which generates input blobs and output blobs for layers.
     * As for the input of each layer used for updating eq_filter_, 
     * use the same input as feed-forward process
     * The layer on the top of the whole network does not has eq_filter_top.
     * Since the feedforward processes the layers in layers_ one by one,
     * the feedback will follow the reverse order
     */
    for (int i = 0; i < this->layers_.size() - 1; ++i) {
      //for each layer, use the successor layer's eq_filter_ as the top_filter
        this->eq_filter_top_.push_back(this->layers_[i+1]->eq_filter());
    }
    this->eq_filter_top_.push_back(NULL);
    //The bottom will adopts the bottom_ vector in the feedforward training process
  }

  template<typename Dtype>
  void FeedbackNet<Dtype>::Visualize(int startLayerIdx, int startChannelIdx, int startOffset, int endLayerIdx){
    //The visualization must be completed by performing Forward() in advance
    if (this->forwardCompleteFlag == false) {
        LOG(ERROR)<<"[Error] Have to input image into the network to complete visualization";
        return;
    }
    if(startLayerIdx < 0 || endLayerIdx < 0){
        LOG(ERROR)<<"Wrong Layer for visualization: "<<startLayerIdx <<"->" <<endLayerIdx;
        return;
    }
    if (startLayerIdx == 0){
        //By default, start from the last layer (fc8 or softmax layer)
      startLayerIdx = this->layers_.size() - 1;
    }

    this->startLayerIdx_ = startLayerIdx;
    this->endLayerIdx_ = endLayerIdx;
    this->startChannelIdx_ = startChannelIdx;
    //0. Construct the layer index array for visualization
    InitVisualization();
    //1. Construct the mask for the input channel
    generateStartTopFilter();
    //2. For each layer in the vector, calculate the eq_filter_ for each layer
    UpdateEqFilter();
    //3. Finally, get the eq_filter_ at the end layer as the output;
    Blob<Dtype>* eq_filter_output  = this->eq_filter_top_[this->endLayerIdx_];
    //4. re-organize the eq_filter_output to get this->visualization_
    Blob<Dtype>* input_blob = (this->bottom_vecs_[endLayerIdx])[0];
    this->visualization_ = new Blob<Dtype>(input_blob->num(), input_blob->channels(),
        input_blob->height(), input_blob->width());
    Dtype* visualization_data_ptr = this->visualization_->mutable_cpu_data();
    Dtype* eq_filter_data_ptr = eq_filter_output->mutable_cpu_data();
    //4.1 Generate the visualization mask, given the startOffset
    int input_length = eq_filter_output->width();
    int output_length = eq_filter_output->height();
    Dtype* _visualization_mask = new Dtype[output_length];
    memset(_visualization_mask, 0, sizeof(Dtype) * output_length);
    if (startOffset == -1){
        //use all the outputs to visualize
        for(int i = 0; i<output_length; ++i) {
            _visualization_mask[i] = (Dtype)1.;
        }
    }
    else {
        _visualization_mask[startOffset] = (Dtype) 1.;
    }
    //4.2 Inner product
    for (int n = 0; n < this->visualization_->num(); ++n) {
        //using caffe_cpu_gemm() function

        caffe::caffe_cpu_gemm(CblasTrans, CblasNoTrans, 1, input_length, output_length,
                              (Dtype)1., eq_filter_data_ptr,
                              _visualization_mask, (Dtype)0.,
                              visualization_data_ptr);
        eq_filter_data_ptr += eq_filter_output->offset(1,0);
        visualization_data_ptr += this->visualization_->offset(1,0);
    }
    //4.3 Reshape the visualization - nothing to do
  }

  template<typename Dtype>
  void FeedbackNet<Dtype>::Visualize(string startLayer, int startChannelIdx, int startOffset, string endLayer){
    int startLayerIdx = this->layer_names_index_[startLayer];
    int endLayerIdx = this->layer_names_index_[endLayer];

    this->Visualize(startLayerIdx, startChannelIdx, startOffset, endLayerIdx);
  }

  template<typename Dtype>
  void FeedbackNet<Dtype>::generateStartTopFilter(){
    //Generate the top_filter_ for the start layer of visualization
    Blob<Dtype>* _top_output_blob = (this->top_vecs_[startLayerIdx_])[0];
    int _input_num = _top_output_blob->channels() * _top_output_blob->height() * _top_output_blob->width();
    int _output_num = _top_output_blob->height() * _top_output_blob->width();
    int _img_num = (this->bottom_vecs_[0])[0]->num();
    start_top_filter_ = new Blob<Dtype>(_img_num, 1, _output_num, _input_num);

    //Initialize the value of start_top_filter_
    Dtype* start_top_filter_data = this->start_top_filter_->mutable_cpu_data();
    int channel_offset = _output_num;

    memset(start_top_filter_data, 0, sizeof(Dtype) * start_top_filter_->count());

    for (int n = 0; n<start_top_filter_->num(); n++) {
        for(int i = 0; i<_output_num; i++) {
            *(start_top_filter_data + channel_offset * this->startChannelIdx_ + i * _input_num + i) = Dtype(1.);
        }
        start_top_filter_data += start_top_filter_->offset(1, 0);
    }
  }

  template<typename Dtype>
  void FeedbackNet<Dtype>::UpdateEqFilter(){
    for(int i = this->startLayerIdx_; i > this->endLayerIdx_; --i){
        //Perform UpdataEqFilter()
        if (i == this->startLayerIdx_){
            this->layers_[i]->UpdateEqFilter(this->start_top_filter_, this->bottom_vecs_[i]);
        }
        else{
            this->layers_[i]->UpdateEqFilter(this->eq_filter_top_[i], this->bottom_vecs_[i]);
        }
    }
  }

  template<typename Dtype>
  void FeedbackNet<Dtype>::DrawVisualization(string dir) {
    std::ostringstream convert;
    for (int n = 0; n<visualization_->num(); n++) {
        convert<<dir<<n;
        string filename = convert.str();
        if (visualization_->channels() == 3 || visualization_->channels() == 1){
            //Visualize as RGB / Grey image
            filename = filename + ".jpg";
            const int _height = visualization_->height();
            const int _width = visualization_->width();
            const int _channel = visualization_->channels();
            WriteDataToImage(filename, _channel, _height, _width,
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
                WriteDataToImage(filename, _channel, _height, _width,
                    visualization_->mutable_cpu_data() + visualization_->offset(n, c) );
            }
        }
    }
  }

  INSTANTIATE_CLASS(FeedbackNet);

} //namespace caffe
