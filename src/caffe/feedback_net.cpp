// Copyright Xianming Liu, July 2014.

#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layer.hpp"
#include "caffe/feedback_layer.hpp"
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

}

template<typename Dtype>
void FeedbackNet<Dtype>::Visualize(int startLayerIdx, int startChannelIdx, int endLayerIdx){
	if(startLayerIdx < 0 || endLayerIdx < 0){
		LOG(ERROR)<<"Wrong Layer for visualization: "<<startLayerIdx <<"->" <<endLayerIdx;
		return;
	}
	if (startLayerIdx == 0)
		startLayerIdx = this->layers_.size() - 1;
	//0. Construct the layer index array for visualization
	//1. Construct the mask for the input channel (if it is convolutional layers)
	//2. For each layer in the vector, calculate the eq_filter_ for each layer
	//3. Finally, get the eq_filter_ at the end layer as the output;
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
