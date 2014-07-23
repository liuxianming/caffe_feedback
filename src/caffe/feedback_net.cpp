// Copyright Xianming Liu, July 11 2014.

#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <cblas.h>

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
      //actually, in this stage, all the ptrs are NULL pointer
        this->eq_filter_top_.push_back(this->layers_[i+1]->eq_filter());
    }
    this->eq_filter_top_.push_back(NULL);
    //The bottom will adopts the bottom_ vector in the feedforward training process
  }

  template<typename Dtype>
  Blob<Dtype>* FeedbackNet<Dtype>::VisualizeSingleNeuron(int startLayerIdx,
      int startChannelIdx, int startOffset) {
    //1. Construct the mask for the input channel
    generateStartTopFilter(startOffset);
    //2. For each layer in the vector, calculate the eq_filter_ for each layer
    UpdateEqFilter();
    //3. Finally, get the eq_filter_ at the data layer as the output;
    Blob<Dtype>* eq_filter_output  = this->eq_filter_top_[0];
    //4. re-organize the eq_filter_output to get this->visualization_
    Blob<Dtype>* input_blob = (this->blobs_[0]).get();
    Blob<Dtype>* _visualization = new Blob<Dtype>(input_blob->num(), input_blob->channels(),
        input_blob->height(), input_blob->width());
    Dtype* visualization_data_ptr = _visualization->mutable_cpu_data();
    Dtype* eq_filter_data_ptr = eq_filter_output->mutable_cpu_data();
    //Copy data
    memcpy(visualization_data_ptr, eq_filter_data_ptr, sizeof(Dtype)*_visualization->count());
    return _visualization;
  }

  template<typename Dtype>
  void FeedbackNet<Dtype>::Visualize(int startLayerIdx, int startChannelIdx, int startOffset, bool test_flag){
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
    //for debug
    if(test_flag) {
        LOG(INFO)<<"Testing feedback filter calculation from ["<<this->layers_[startLayerIdx_]->layer_param().name()
            <<"] to [Data Layer]";
    }
    else{
        LOG(INFO)<<"Visualization from ["<<this->layers_[startLayerIdx_]->layer_param().name()
            <<"] to [Data Layer]";
    }
    //0. Construct the layer index array for visualization
    InitVisualization();
    if(startOffset >= 0) {
        this->visualization_ = VisualizeSingleNeuron(startLayerIdx_, startChannelIdx_, startOffset);
        if(test_flag){
            //output the errors
            Dtype error = 0;
            for(int n = 0; n<this->visualization_->num();++n){
                Dtype* img_data_ptr = this->blobs_[0]->mutable_cpu_data() + this->blobs_[0]->offset(n);
                Dtype* filter_ptr = this->visualization_->mutable_cpu_data() + this->visualization_->offset(n);
                int len = this->visualization_->width() * this->visualization_->height();
                Dtype predicted_output = caffe_cpu_dot<Dtype>(len, img_data_ptr, filter_ptr);
                Dtype output_val = *((this->top_vecs_[this->startLayerIdx_])[0]->mutable_cpu_data()
                    + (this->top_vecs_[this->startLayerIdx_])[0]->offset(n, this->startChannelIdx_)
                    + startOffset);
                LOG(INFO)<<"[Predicted value for neuron #"<<startOffset<<"] = "<<predicted_output<<" / "<<output_val;
                error += (output_val - predicted_output) * (output_val - predicted_output);
                LOG(INFO)<<"[ERROR for neuron #"<<startOffset<<"] = "<<error;
            }
        }
    }
    else if(startOffset == -1){
        //Visualize the neurons of the whole channels
        //especially designed for convolutional layers
        //Create visualization variable
        Blob<Dtype>* input_blob = (this->blobs_[0]).get();
        Blob<Dtype>* _visualization = new Blob<Dtype>(input_blob->num(), input_blob->channels(),
            input_blob->height(), input_blob->width());
        memset(_visualization->mutable_cpu_data(), 0, sizeof(Dtype)*_visualization->count());

        Blob<Dtype>* _filter_output = (this->top_vecs_[startLayerIdx_])[0];
        int _output_num = _filter_output->height() * _filter_output->width();
        for(int index = 0; index < _output_num; ++index){
            //Plot information every 100 neurons
            if(index%10 == 0) {
                LOG(INFO)<<"Processing "<<index<<" neurons";
            }
            Blob<Dtype>* _single_visualization =
                VisualizeSingleNeuron(startLayerIdx_, startChannelIdx_, index);
            //Add the values together
            for(int n = 0; n<_single_visualization->num(); ++n){
                Dtype _output_value = *(_filter_output->mutable_cpu_data() +
                    _filter_output->offset(n, startChannelIdx_) + index);
                Dtype* _single_visualization_ptr = _single_visualization->mutable_cpu_data() + _single_visualization->offset(n);
                Dtype* _visualization_ptr = _visualization->mutable_cpu_data() + this->visualization_->offset(n);
                caffe_cpu_vsum<Dtype>(_single_visualization->width()*_single_visualization->height(),
                    _output_value, _single_visualization_ptr,
                    (Dtype)1., _visualization_ptr);
            }
        }
        this->visualization_ = _visualization;
    }

    //Normalization:
    for(int n = 0; n<this->visualization_->num(); ++n) {
        for (int c = 0; c<this->visualization_->channels(); c++) {
            ImageNormalization<Dtype>(this->visualization_->mutable_cpu_data() + this->visualization_->offset(n, c),
                this->visualization_->offset(0,1), (Dtype)128);
        }
    }
  }

  template<typename Dtype>
  void FeedbackNet<Dtype>::Visualize(string startLayer, int startChannelIdx, int startOffset, bool test_flag){
    int startLayerIdx = this->layer_names_index_[startLayer];

    this->Visualize(startLayerIdx, startChannelIdx, startOffset, test_flag);
  }

  template<typename Dtype>
  void FeedbackNet<Dtype>::generateStartTopFilter(int startOffset){
    //Generate the top_filter_ for the start layer of visualization
    //The output of the top_filter is only one neuron,
    Blob<Dtype>* _top_output_blob = (this->top_vecs_[startLayerIdx_])[0];
    long _input_num = _top_output_blob->channels() * _top_output_blob->height() * _top_output_blob->width();
    long channel_offset = _top_output_blob->height() * _top_output_blob->width();
    long _img_num = this->blobs_[0]->num();

    //    size_t tempsize = _img_num * _input_num * _output_num;
    //    LOG(INFO)<<_input_num<<"*"<<_output_num<<"*"<<_img_num<<"="<<tempsize;

    start_top_filter_ = new Blob<Dtype>(_img_num, 1, 1, _input_num);

    //Initialize the value of start_top_filter_
    Dtype* start_top_filter_data = start_top_filter_->mutable_cpu_data();

    memset(start_top_filter_data, 0, sizeof(Dtype) * start_top_filter_->count());

    for (int n = 0; n<start_top_filter_->num(); n++) {
        *(start_top_filter_data + channel_offset * this->startChannelIdx_ + startOffset) = Dtype(1.);
        start_top_filter_data += start_top_filter_->offset(1, 0);
    }
}

  template<typename Dtype>
  void FeedbackNet<Dtype>::UpdateEqFilter(){
    //firs build top_filter vector
    this->eq_filter_top_[this->startLayerIdx_] = this->start_top_filter_;
    for(int i = this->startLayerIdx_; i > 0; --i){
        //The 0-th layer is data layer, no need to perform UpdateEqFilter()
        //Perform UpdataEqFilter()
        this->layers_[i]->UpdateEqFilter(this->eq_filter_top_[i], this->bottom_vecs_[i]);
        //Update pointer
        this->eq_filter_top_[i-1] = this->layers_[i]->eq_filter();
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
