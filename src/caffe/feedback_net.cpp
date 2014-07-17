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

using std::pair;
using std::map;
using std::set;
using std::string;

namespace caffe{

  template<typename Dtype>
  void FeedbackNet<Dtype>::InitVisualization(){
    //copy the ptr of layers to fLayers
    //The first layer is not included, because it is input data layer
    for(int i = 1; i < this->layers_.size(); i++){
        shared_ptr<Layer<Dtype> > _layer = this->layers_[i];
        fLayers_.push_back(_layer);
    }

    //setup the eq_filter_ flow 
    /*
     * Follow the similar work flow as in Net.Init(),
     * which generates input blobs and output blobs for layers.
     * As for the input of each layer used for updating eq_filter_, 
     * use the same input as feed-forward process
     * Remember here, fLayerIndex = LayerIndex - 1, which omits the input data layer
     * and the layer on the top of the whole network does not has eq_filter_top
     */
    for (int i = 0; i < fLayers_.size() - 1; ++i) {
      //LayerParameter layer_param = fLayers_[i]->layer_param_;
      //int output_layer_idx = this->layer_names_index_[layer_param.layer_name];
    }

  }

  template<typename Dtype>
  void FeedbackNet<Dtype>::Visualize(int startLayerIdx, int startChannelIdx, int endLayerIdx){
    if(startLayerIdx < 0 || endLayerIdx < 0){
        LOG(ERROR)<<"Wrong Layer for visualization: "<<startLayerIdx <<"->" <<endLayerIdx;
        return;
    }
    if (startLayerIdx == 0)
      startLayerIdx = this->layers_.size() - 1;

    //reduce 1: because the layers include input layer,
    //which has no feedback attributes (e.g., data, cifar, etc)
    //and will not be counted in the fLayers_
    this->startLayerIdx_ = startLayerIdx - 1;
    this->endLayerIdx_ = endLayerIdx - 1;
    this->startChannelIdx_ = startChannelIdx;
    //0. Construct the layer index array for visualization
    InitVisualization();
    //1. Construct the mask for the input channel (if it is convolutional layers)
    //2. For each layer in the vector, calculate the eq_filter_ for each layer
    UpdateEqFilter();
    //3. Finally, get the eq_filter_ at the end layer as the output;
    Blob<Dtype>* eq_filter  = fLayers_[endLayerIdx_]->eq_filter();
    //Finally, re-organize the eq_filter to get this->visualization_

  }

  template<typename Dtype>
  void FeedbackNet<Dtype>::Visualize(string startLayer, int startChannelIdx, string endLayer){
    int startLayerIdx = this->layer_names_index_[startLayer];
    int endLayerIdx = this->layer_names_index_[endLayer];

    this->Visualize(startLayerIdx, startChannelIdx, endLayerIdx);
  }

  template<typename Dtype>
  void FeedbackNet<Dtype>::UpdateEqFilter(){

  }

  template<typename Dtype>
  void FeedbackNet<Dtype>::DrawVisualization(string dir) {
    if (visualization_.size() == 0) {
        LOG(ERROR)<<"[------------Visualization Error: No Input------------]";
        return;
    }
    std::ostringstream convert;
    for (int s = 0; s<visualization_.size(); s++){
        for (int n = 0; n<visualization_[s]->num(); n++) {
            convert<<dir<<s<<"_"<<n;
            string filename = convert.str();
            if (visualization_[s]->channels() == 3 || visualization_[s]->channels() == 1){
                //Visualize as RGB / Grey image
                filename = filename + ".jpg";
                const int _height = visualization_[s]->height();
                const int _width = visualization_[s]->width();
                const int _channel = visualization_[s]->channels();
                WriteDataToImage(filename, _channel, _height, _width,
                    visualization_[s]->mutable_cpu_data() + visualization_[s]->offset(n) );
            }
            else{
                //Visualize each channel as a separate image
                for(int c = 0; c<visualization_[s]->channels(); c++) {
                    convert<<filename<<"_"<<c<<".jpg";
                    string filename = convert.str();
                    const int _height = visualization_[s]->height();
                    const int _width = visualization_[s]->width();
                    const int _channel = visualization_[s]->channels();
                    WriteDataToImage(filename, _channel, _height, _width,
                        visualization_[s]->mutable_cpu_data() + visualization_[s]->offset(n, c) );
                }
            }
        }
    }
  }

  INSTANTIATE_CLASS(FeedbackNet);

} //namespace caffe
