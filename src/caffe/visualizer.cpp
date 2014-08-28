// Copyright 2014 Xianming Liu

#include <cstdio>
#include <iostream>
#include <boost/filesystem.hpp>

#include <algorithm>
#include <string>
#include <vector>
#include <sstream>

#include "caffe/net.hpp"
#include "caffe/feedback_net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/visualizer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;
using std::min;

namespace caffe {
  template<typename Dtype>
  Visualizer<Dtype>::Visualizer(const VisualizerParameter& param)
    : net_() {
    Init(param);
  }

  template <typename Dtype>
  Visualizer<Dtype>::Visualizer(const string& param_file)
    : net_() {
    VisualizerParameter param;
    ReadProtoFromTextFile(param_file, &param);
    Init(param);
  }

  template<typename Dtype>
  void Visualizer<Dtype>::Init(const VisualizerParameter& param) {
    param_ = param;
    //Set mode
    Caffe::set_mode(Caffe::Brew(param_.visualizer_mode()));
    if (param_.visualizer_mode() == VisualizerParameter_VisualizerMode_GPU && 
	param_.has_device_id()) {
      Caffe::SetDevice(param_.device_id());
    }
    //Initialize net_
    LOG(INFO) << "Creating net used for visualization.";
    net_.reset(new FeedbackNet<Dtype>(param_.visualization_net()));
    //Load model
    LOG(INFO)<<"Load model from "<<param_.model();
    (net_.get())->CopyTrainedLayersFrom(param_.model());
    //Output visualization task summerization
    LOG(INFO)<<"**************************************************";
    LOG(INFO)<<"Job summerization:";
    LOG(INFO)<<"Visualize from Layer ["<<param_.target_layer()<<"] ";
    if (param_.visualization_type() == VisualizerParameter_VisualizationType_TOP){
      //visualize using top k neurons
      LOG(INFO)<<"Using Top "<<param_.k()<<" neuron(s)";
    }
    else if (param_.visualization_type() == VisualizerParameter_VisualizationType_SELECT){
      LOG(INFO)<<"Using given neurons: "
	       <<" Channel = "<<param_.channel_idx()
	       <<" Row = "<<param_.row_idx()
	       <<" Col = "<<param_.col_idx();
    }
    LOG(INFO)<<"Max Iteration = "<<param_.max_iter();
    LOG(INFO)<<"Save Path: "<<param_.store_dir();
    LOG(INFO)<<"**************************************************";
  }

  template<typename Dtype>
  void Visualizer<Dtype>::Visualize(){
    LOG(INFO)<<"Start Visualization";
    //Create dir
    boost::filesystem::path save_dir((param_.store_dir()).c_str());
    if(boost::filesystem::create_directories(save_dir)) {
      LOG(INFO)<<"Create saving dir "<<param_.store_dir();
    }
    //need to store the original input
    if(param_.original_output()) {
      LOG(INFO)<<"[Visualizer Job]: Need to visualize the original input images";
      boost::filesystem::path original_save_dir((param_.original_store_dir()).c_str());
      if(boost::filesystem::create_directories(original_save_dir)) {
        LOG(INFO)<<"Create original input image saving dir "<<param_.original_store_dir();
      }
    }
    //Start iteration
    for(iter_ = 0; iter_ < param_.max_iter(); ++iter_){
      LOG(INFO)<<"Processing Batch "<<iter_;
      //Perform forward()
      if(param_.visualization_type() == VisualizerParameter_VisualizationType_TOP){
        if(param_.visualize_with_feedback()) {
          LOG(INFO) << "visualize with feedback";
          const vector<Blob<Dtype>*>& result = net_->FeedbackForwardPrefilled();
        } else {
          LOG(INFO) << "visualize with feedforward";
          const vector<Blob<Dtype>*>& result = net_->ForwardPrefilled();
        }
        LOG(INFO)<<"Forwarding function complete";
        //Start visualization
        net_->VisualizeTopKNeurons(param_.target_layer(), param_.k(), false);
      }
      else if (param_.visualization_type() == VisualizerParameter_VisualizationType_SELECT) {
        //const vector<Blob<float>*>& result = cnet_.FeedbackForwardPrefilled();
        const vector<Blob<Dtype>*>& result = net_->ForwardPrefilled();
        net_->Visualize(param_.target_layer(), param_.channel_idx(), param_.row_idx(), param_.col_idx(), false);
      }
      //Save visualization to image files
      std::ostringstream convert;
      convert << iter_ <<"_";
      string prefix = convert.str();
      net_->DrawVisualization(param_.store_dir(), prefix, 1.0);

      //need to visualize the original input image
      if (param_.original_output()) {
        LOG(INFO)<<"Saving original input images to files...";
        net_->DrawInputImages(param_.original_store_dir(), prefix, 1.0);
      }
      LOG(INFO)<<"Complete";
    }
  }

  // The function is used to visualize the neurons with the images 
  // which response most significantly at the given neuron
  template<typename Dtype>
  void Visualizer<Dtype>::VisualizeNeurons(){
    LOG(INFO)<<"Start Visualization along neurons";
    //Create dir
    boost::filesystem::path save_dir((param_.store_dir()).c_str());
    if(boost::filesystem::create_directories(save_dir)) {
      LOG(INFO)<<"Create saving dir "<<param_.store_dir();
    }
    //Start iteration
    for(iter_ = 0; iter_ < param_.max_iter(); ++iter_){
      LOG(INFO)<<"Processing Batch "<<iter_;
      //Perform forward()
      const vector<Blob<Dtype>*>& result = net_->FeedbackForwardPrefilled();      
      LOG(INFO)<<"Forwarding function complete";
      //Start visualization
      vector<int*> channel_offsets = net_->VisualizeTopKNeurons(param_.target_layer(), param_.k(), false);

      //Save visualization to image files
      std::ostringstream convert;
      convert << iter_ <<"_";
      string prefix = convert.str();
      net_->DrawNeron(param_.store_dir(), channel_offsets[0], prefix, 1.0);
      LOG(INFO)<<"Complete";
    }
  }

  INSTANTIATE_CLASS(Visualizer);
} // namespace caffe
