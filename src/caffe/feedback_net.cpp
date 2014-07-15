// Copyright Xianming Liu, July 2014.

#include <map>
#include <set>
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

namespace caffe{

template<typename Dtype>
void FeedbackNet<Dtype>::InitVisualization(){

}

template<typename Dtype>
void FeedbackNet<Dtype>::Visualize(int startLayerIdx, int startChannelIdx, int endLayerIdx){

}

template<typename Dtype>
void FeedbackNet<Dtype>::Visualize(string startLayer, int startChannelIdx, string endLayer){

}

template<typename Dtype>
void FeedbackNet<Dtype>::UpdateEqFilter(){

}

INSTANTIATE_CLASS(FeedbackNet);

} //namespace caffe
