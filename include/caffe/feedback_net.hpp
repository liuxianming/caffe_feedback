// Copyright Xianming Liu, July 2014.

#ifndef CAFFE_FEEDBACK_NET_HPP_
#define CAFFE_FEEDBACK_NET_HPP_

#include <map>
#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{

  template <typename Dtype>
  class FeedbackNet : public Net<Dtype>
  {
  public:
    explicit FeedbackNet(const NetParameter& param) : Net<Dtype>(param){}
    explicit FeedbackNet(const string& param_file) : Net<Dtype>(param_file){}
    virtual ~FeedbackNet(){}

    void UpdateEqFilter();

    void UpdateEqFilter(int startLayerIdx, 
			int startChannelIdx, 
			int startOffset);

    void SearchTopKNeurons(int startLayerIdx, int k, int* channel_offset, int* in_channel_offset);

    // By default, the visualization results are straight to the input layer
    // (input_blobs_ as in the father class: Net)
    // Default parameter:
    // startChannelIdx = 0
    // startOffset: visualize the neuron at specific position (pos = startOffset)
    //  if startOffset == -1 (by default), use all the output neurons with the output value as weights.
    void Visualize(int startLayerIdx, int startChannelIdx = 0, int heightOffset = -1, int widthOffset = -1,  bool test_flag = false);
    // Search the layer list and get the index
    void Visualize(string startLayer, int startChannelIdx = 0, int heightOffset  = -1, int widthOffset = -1,  bool test_flag = false);

    void VisualizeTopKNeurons(int startLayerIdx, int k = 1, bool weight_flag = true);
    void VisualizeTopKNeurons(string startLayer, int k = 1, bool weight_flag = true);
    inline Blob<Dtype>* GetVisualization() {return visualization_;}

    //draw visualization: draw visualization_ to files, stored in dir
    void DrawVisualization(string dir, string prefix = "");

    //Member function
    void InitVisualization();

    const vector<Blob<Dtype>*>& FeedbackForwardPrefilled(Dtype* loss = NULL, int startLayerIdx = -1, int channel = -1, int offset = -1);
    const vector<Blob<Dtype>*>& FeedbackForwardPrefilled(string startLayer, int channel = -1, int offset = -1);

  protected:
    //void InitFeedback();

    //Generate the top_filter_ for the startLayer of visualization
    void generateStartTopFilter(int startOffset);
    //Test the eq_filter calculation
    Dtype test_eq_filter(int _layer_idx, Blob<Dtype>* eq_filter);

  protected:
    //Visualize the response of single neuron, and the return value is the visualization results.
    Blob<Dtype>* VisualizeSingleNeuron(int startLayerIdx, int startChannelIdx, int startOffset, bool weigh_flag);
    Blob<Dtype>* VisualizeSingleRowNeurons(int startLayerIdx, int startChannelIdx, int heightOffset);
    Blob<Dtype>* VisualizeSingleColNeurons(int startLayerIdx, int startChannelIdx, int widthOffset);
    //Stores the visualization results
    //num: input images of minibatch;
    //channels, width, height: size of image (channel by default is rgb)
    Blob<Dtype>* visualization_;
    
    //The vector stores the ptr list of eq_filter for each feedback_layer in fLayers_
    //If the end layer idx of visualization is k,
    //then the visualization results are eq_filter_top_[k]
    vector<Blob<Dtype>* > eq_filter_top_;

    int startLayerIdx_;
    int startChannelIdx_;
    int endLayerIdx_;
    int startOffset_;

    //The synthesized top_filter for the start layer of visualization
    Blob<Dtype>* start_top_filter_;
  };
} // namespace caffe

#endif
