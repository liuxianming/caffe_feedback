// Copyright Xianming Liu, July 2014.

#ifndef CAFFE_FEEDBACK_NET_HPP_
#define CAFFE_FEEDBACK_NET_HPP_

#include <map>
#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/feedback_layer.hpp"
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

	// By default, the visualization results are straight to the input layer
	// (input_blobs_ as in the father class: Net)
	// Default parameter:
	// startChannelIdx = -1 : using all the channels of the given layer
	// endLayerIdx = 0: using the data layer as visualization target
	void Visualize(int startLayerIdx, int startChannelIdx = -1, int endLayerIdx = 0);

	// Search the layer list and get the index
	void Visualize(string startLayer, int startChannelIdx = -1, string endLayer = "data");

	inline vector<Blob<Dtype>* > GetVisualization() {return visualization_;}

	//draw visualization: draw visualization_ to files, stored in dir
	void DrawVisualization(string dir);

protected:
	//Member function
	void InitVisualization();
	//void InitFeedback();

protected:
	//The vector stores the visualization results
	//vector: input source;
	//num: input images of minibatch;
	//channels, width, height: size of image (channel by default is rgb)
	vector<Blob<Dtype>* > visualization_;
	int startLayerIdx_;
	int startChannelIdx_;
	int endLayerIdx_;

	vector<shared_ptr<FeedbackLayer<Dtype> > > feedback_layers_;
};
} // namespace caffe

#endif
