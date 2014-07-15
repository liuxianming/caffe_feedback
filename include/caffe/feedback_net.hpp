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
	explicit FeedbackNet(const NetParameter& param) : Net<Dtype>(param);
	explicit FeedbackNet(const string& param_file) : Net<Dtype>(param_file);
	virtual ~FeedbackNet(){}

	void UpdateEqFilter();



protected:
	//Design data member
};
} // namespace caffe

#endif
