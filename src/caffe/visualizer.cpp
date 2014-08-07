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
#include "caffe/visualier.hpp"
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
		net_.reset(new Net<Dtype>(param_.visualization_net()));
		//Load model
		LOG(INFO)<<"Load model from "<<param_.model();
		(net_.get())->CopyTrainedLayersFrom(param_.model());
		//Output visualization task summerization
		LOG(INFO)<<"**************************************************";
		LOG(INFO)<<"Job summerization:"
		LOG(INFO)<<"Visualize from Layer ["<<param_.target_layer()<<"] "
				 <<"Using Top "<<param_.k()<<" neuron(s)";
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
		//Start iteration
		for(iter_ = 0; iter_ < param_.max_iter(); ++iter_){
			LOG(INFO)<<"Processing Batch "<<iter_;
			//Perform forward()
			const vector<Blob<float>*>& result = caffe_test_net.FeedbackForwardPrefilled();
			//Start visualization
			caffe_test_net.VisualizeTopKNeurons(param_.target_layer(), param_.k(), false);
			//Save visualization to image files
			std::ostringstream convert;
			convert << iter_ <<"_";
			string prefix = convert.str();
			net_.DrawVisualization(param_.store_dir(), prefix);
			LOG(INFO)<<"Complete";
		}
	}

INSTANTIATE_CLASS(Visualizer)
} // namespace caffe