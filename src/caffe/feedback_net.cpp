// Copyright Xianming Liu, July 11 2014.

#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

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
  }

  template<typename Dtype>
  Blob<Dtype>* FeedbackNet<Dtype>::VisualizeSingleNeuron(int startLayerIdx, 
							 int startChannelIdx, 
							 int startOffset, 
							 bool weight_flag){
    Blob<Dtype>* input_blob = (this->blobs_[0]).get();
    int num = input_blob->num(); 
    int* startChannelIdxs = new int[num];
    int* startOffsets = new int[num];
    for(int n = 0; n<num; ++n){
      startChannelIdxs[n] = startChannelIdx;
      startOffsets[n] = startOffset;
    }
    VisualizeSingleNeuron(startLayerIdx, startChannelIdxs, startOffsets, weight_flag);
  }

  template<typename Dtype>
  Blob<Dtype>* FeedbackNet<Dtype>::VisualizeSingleNeuron(int startLayerIdx,
							 int* startChannelIdxs, 
							 int* startOffsets, 
							 bool weight_flag) {
    UpdateEqFilter(startLayerIdx, startChannelIdxs, startOffsets);
    //Get the eq_filter_ at the data layer as the output;
    Blob<Dtype>* eq_filter_output  = this->eq_filter_top_.back();
    //Re-organize the eq_filter_output to get this->visualization_
    Blob<Dtype>* input_blob = (this->blobs_[0]).get();
    Blob<Dtype>* _visualization 
      = new Blob<Dtype>(input_blob->num(), input_blob->channels(),
			input_blob->height(), input_blob->width());
    memcpy(_visualization->mutable_cpu_data(), eq_filter_output->cpu_data(), 
	   sizeof(Dtype)*eq_filter_output->count());
    //Find weight
    Blob<Dtype>* _filter_output = (this->top_vecs_[startLayerIdx_])[0];
    Dtype* output_weights = new Dtype[_filter_output->num()];
    for(int i = 0; i<_filter_output->num(); ++i) {
      output_weights[i] = *(_filter_output->mutable_cpu_data() 
			    + _filter_output->offset(i, startChannelIdxs[i]) + startOffsets[i]);
    }
    if(weight_flag) {
      _visualization->multiply( output_weights);
    }
    return _visualization;
  }

  template<typename Dtype>
  void FeedbackNet<Dtype>::Visualize(int startLayerIdx, int startChannelIdx, 
				     int heightOffset, int widthOffset, bool test_flag){
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
    
    //0. Construct the layer index array for visualization
    InitVisualization();
    int output_layer_width = (this->top_vecs_[startLayerIdx_])[0]->width();
    int output_layer_height = (this->top_vecs_[startLayerIdx_])[0]->height();
    int output_layer_channel = (this->top_vecs_[startLayerIdx_])[0]->channels();
    //LOG(INFO)<<output_layer_width<<":"<<output_layer_height<<":"<<output_layer_channel;
    if(heightOffset >= 0 && widthOffset >= 0) {

      this->startOffset_ = widthOffset + heightOffset * output_layer_width;
      if (this->startOffset_ >= (this->top_vecs_[startLayerIdx])[0]->width() * (this->top_vecs_[startLayerIdx_])[0]->height()) {
	LOG(ERROR)<<"Wrong output neuron!";
	return;
      }
      this->visualization_ = VisualizeSingleNeuron(startLayerIdx_, startChannelIdx_, this->startOffset_, false);
    }
    else if(heightOffset >= 0 && widthOffset < 0){
      //visualize single row
      this->visualization_ = VisualizeSingleRowNeurons(startLayerIdx, startChannelIdx, heightOffset);
    }
    else if(heightOffset < 0 && widthOffset >= 0){
      //visualize single col
      this->visualization_ = VisualizeSingleColNeurons(startLayerIdx, startChannelIdx, widthOffset);
    }
    else {
      Blob<Dtype>* input_blob = (this->blobs_[0]).get();
      Blob<Dtype>* _visualization = new Blob<Dtype>(input_blob->num(), input_blob->channels(),
						    input_blob->height(), input_blob->width());
      memset(_visualization->mutable_cpu_data(), 0, sizeof(Dtype)*_visualization->count());

      for(int h=0; h<output_layer_height; ++h) {
	_visualization->add( *(VisualizeSingleRowNeurons(startLayerIdx, startChannelIdx, h)));
      }
      this->visualization_ = _visualization;
    }
  }

  template<typename Dtype>
  void FeedbackNet<Dtype>::Visualize(string startLayer, int startChannelIdx, 
				     int heightOffset, int widthOffset, 
				     bool test_flag){
    int startLayerIdx = this->layer_names_index_[startLayer];

    this->Visualize(startLayerIdx, startChannelIdx, heightOffset, widthOffset,  test_flag);
  }

  template<typename Dtype>
  void FeedbackNet<Dtype>::SearchTopKNeurons(int startLayerIdx, int k, 
					     vector<int*> channel_offsets, vector<int*> in_channel_offsets){
    Blob<Dtype>* output_blobs = this->top_vecs_[startLayerIdx][0];
    int output_blob_channel_size = output_blobs->width() * output_blobs->height();
    for(int n = 0; n<output_blobs->num(); ++n){
      Dtype* output_blobs_data = output_blobs->mutable_cpu_data()+output_blobs->offset(n);
      //sort the output for the first image in mini-batch
      vector<Dtype> sorted_vector(output_blobs_data, output_blobs_data 
				  + output_blobs->count()/output_blobs->num());
      std::sort(sorted_vector.begin(), sorted_vector.end());
      //get the top k
      typename vector<Dtype>::iterator pos = sorted_vector.end();
      for(int i = 0; i<k; ++i){
	pos --; 
	Dtype _val = *pos;
	Dtype* p = std::find(output_blobs_data, output_blobs_data + output_blobs->count(), _val);
	int offset = p - output_blobs_data;
	(channel_offsets[i])[n] = offset / output_blob_channel_size;
	(in_channel_offsets[i])[n] = offset % output_blob_channel_size;
      }
    }
  }

  template<typename Dtype>
  void FeedbackNet<Dtype>::VisualizeTopKNeurons(string startLayer, int k, bool weight_flag){
    int startLayerIdx = this->layer_names_index_[startLayer];
    this->VisualizeTopKNeurons(startLayerIdx, k);
  }

  template<typename Dtype>
  void FeedbackNet<Dtype>::VisualizeTopKNeurons(int startLayerIdx, int k, bool weight_flag){
    this->startLayerIdx_ = startLayerIdx;
    Blob<Dtype>* input_blob = (this->blobs_[0]).get();
    Blob<Dtype>* _visualization 
      = new Blob<Dtype>(input_blob->num(), input_blob->channels(),
			input_blob->height(), input_blob->width());
    memset(_visualization->mutable_cpu_data(), 0, sizeof(Dtype)*_visualization->count());

    //each int* corresponds to a rank, sized in the number of images in each batch
    vector<int*> channel_offsets;
    vector<int*> in_channel_offsets;
    for(int i = 0; i < k; ++i){
      int* channel_offset = new int[input_blob->num()];
      channel_offsets.push_back(channel_offset);
      int* in_channel_offset = new int[input_blob->num()];
      in_channel_offsets.push_back(in_channel_offset);
    }
    
    SearchTopKNeurons(startLayerIdx, k, channel_offsets, in_channel_offsets);
    for(int i = 0; i<k; ++i){
      int* channel_offset = channel_offsets[i];
      int* in_channel_offset = in_channel_offsets[i];
      for(int n = 0; n<input_blob->num(); ++n){
	LOG(INFO)<<"Visualize image "<< n <<" using "<< channel_offset[n] <<" / "<<in_channel_offset[n];
      }
      _visualization ->add(*(VisualizeSingleNeuron(startLayerIdx, 
							channel_offset, 
							in_channel_offset,
							weight_flag)));
    }
    this->visualization_ = _visualization;
  }

  template<typename Dtype>
  Blob<Dtype>* FeedbackNet<Dtype>::VisualizeSingleRowNeurons(int startLayerIdx, int startChannelIdx, int heightOffset) {
    int output_layer_width = (this->top_vecs_[startLayerIdx])[0]->width();
    int output_layer_height = (this->top_vecs_[startLayerIdx])[0]->height();

    Blob<Dtype>* input_blob = (this->blobs_[0]).get();
    Blob<Dtype>* _visualization 
      = new Blob<Dtype>(input_blob->num(), input_blob->channels(),
			input_blob->height(), input_blob->width());
    memset(_visualization->mutable_cpu_data(), 0, sizeof(Dtype)*_visualization->count());

    if(heightOffset < 0 || heightOffset >= output_layer_height) {
      LOG(ERROR)<<"Wrong Height offset";
    }
    else{
        for (int i = 0; i<output_layer_width; ++i) {
            int startOffset = i + output_layer_width * heightOffset;
            _visualization->add(*(VisualizeSingleNeuron(startLayerIdx, startChannelIdx, 
							startOffset, true)));
        }
    }
    return _visualization;
  }

  template<typename Dtype>
  Blob<Dtype>* FeedbackNet<Dtype>::VisualizeSingleColNeurons(int startLayerIdx, int startChannelIdx, int widthOffset) {
    int output_layer_width = (this->top_vecs_[startLayerIdx])[0]->width();
    int output_layer_height = (this->top_vecs_[startLayerIdx])[0]->height();

    Blob<Dtype>* input_blob = (this->blobs_[0]).get();
    Blob<Dtype>* _visualization 
      = new Blob<Dtype>(input_blob->num(), input_blob->channels(),
			input_blob->height(), input_blob->width());
    memset(_visualization->mutable_cpu_data(), 0, sizeof(Dtype)*_visualization->count());

    if(widthOffset < 0 || widthOffset >= output_layer_width) {
        LOG(ERROR)<<"Wrong Height offset";
    }
    else{
        for (int i = 0; i<output_layer_height; ++i) {
            int startOffset = i + output_layer_width + widthOffset;;
            _visualization->add(*(VisualizeSingleNeuron(startLayerIdx, startChannelIdx, 
							startOffset, true)));
        }
    }
    return _visualization;
  }

  template<typename Dtype>
  void FeedbackNet<Dtype>::generateStartTopFilter(int* startChannelIdxs,  int* startOffsets){
    //Generate the top_filter_ for the start layer of visualization
    //The output of the top_filter is only one neuron,
    Blob<Dtype>* _top_output_blob = (this->top_vecs_[startLayerIdx_])[0];
    long _input_num 
      = _top_output_blob->channels() 
      * _top_output_blob->height()
      * _top_output_blob->width();
    long channel_offset = _top_output_blob->height() * _top_output_blob->width();
    long _img_num = this->blobs_[0]->num();

    start_top_filter_ = new Blob<Dtype>(_img_num, 1, 1, _input_num);

    //Initialize the value of start_top_filter_
    Dtype* start_top_filter_data = start_top_filter_->mutable_cpu_data();

    memset(start_top_filter_data, 0, sizeof(Dtype) * start_top_filter_->count());

    for (int n = 0; n<start_top_filter_->num(); n++) {
      int startChannelIdx = startChannelIdxs[n];
      int startOffset = startOffsets[n];
      *(start_top_filter_data + channel_offset * startChannelIdx + startOffset) 
	= Dtype(1.);
      start_top_filter_data += start_top_filter_->offset(1, 0);
    }
  }


  template<typename Dtype>
  void FeedbackNet<Dtype>::UpdateEqFilter(int startLayerIdx, 
					  int* startChannelIdxs, 
					  int* startOffsets){
    this->startLayerIdx_ = startLayerIdx;
    //Generate top mask
    generateStartTopFilter(startChannelIdxs, startOffsets);
    //Begin updating the eq_filter
    //firs build top_filter vector
    this->eq_filter_top_.clear();
    this->eq_filter_top_.push_back(start_top_filter_);
    for(int i = this->startLayerIdx_; i > 0; --i){
      //LOG(INFO)<<"Processing Layer ["<<this->layers_[i]->layer_param().name()<<"]";
      string blob_name = this->blob_names_[(this->bottom_id_vecs_[i])[0]];
      //LOG(INFO)<<"Using BLOB ["<<blob_name<<"] as input";
      this->layers_[i]->UpdateEqFilter(this->eq_filter_top_.back(), this->bottom_vecs_[i]);
      this->eq_filter_top_.push_back(this->layers_[i]->eq_filter());
      //Testing
      bool test_flag = false;
      if (test_flag) {
	Dtype error = test_eq_filter(i, this->eq_filter_top_.back(), startChannelIdxs, startOffsets);
	LOG(INFO)<<"Error of Layer "<<this->layer_names_[i]<<" is "
		 <<error<<" : "<<((error < (Dtype) 16.) ? "OK" : "FAIL");
      }
    }
  }

  template<typename Dtype>
  void FeedbackNet<Dtype>::DrawVisualization(string dir, string prefix) {
    //Normalization:
    for(int n = 0; n<this->visualization_->num(); ++n) {
      for (int c = 0; c<this->visualization_->channels(); c++) {
	ImageNormalization<Dtype>(this->visualization_->mutable_cpu_data() 
				  + this->visualization_->offset(n, c),
				  this->visualization_->offset(0,1), 
				  (Dtype)100, (Dtype)20.0);
      }
    }
    for (int n = 0; n<visualization_->num(); n++) {
      std::ostringstream convert;
      convert<<dir<<prefix<<n;
      string filename = convert.str();
      LOG(INFO)<<"Saving image "<<filename;
      if (visualization_->channels() == 3 || visualization_->channels() == 1){
	//Visualize as RGB / Grey image
	filename = filename + ".jpg";
	const int _height = visualization_->height();
	const int _width = visualization_->width();
	const int _channel = visualization_->channels();
	WriteDataToImage<Dtype>(filename, _channel, _height, _width,
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
	  WriteDataToImage<Dtype>(filename, _channel, _height, _width,
				  visualization_->mutable_cpu_data() 
				  + visualization_->offset(n, c) );
	}
      }
    }
  }

  template<typename Dtype>
  Dtype FeedbackNet<Dtype>::test_eq_filter(int _layer_idx, Blob<Dtype>* eq_filter, 
					   int* startChannelIdxs, int* startOffsets) {
    Blob<Dtype>* feedforward_output_blob = (this->top_vecs_[this->startLayerIdx_])[0];

    Blob<Dtype>* input_blob = (this->bottom_vecs_[_layer_idx])[0];
    LOG(INFO)<<"Testing using blob "
	     <<this->blob_names_[this->bottom_id_vecs_[_layer_idx][0]] <<" as input";
    LOG(INFO)<<"Size: "
	     <<input_blob->channels() 
	     << " * " << input_blob->height() 
	     << " * "<<input_blob->width();
    Dtype error = (Dtype) 0.;

    for(int n = 0; n<input_blob->num(); ++n) {
      int startChannelIdx = startChannelIdxs[n];
      int startOffset = startOffsets[n];
      Dtype output_val = *(feedforward_output_blob->mutable_cpu_data()
			   + feedforward_output_blob->offset(n, startChannelIdx)
			   + startOffset);
      Dtype* input_data = input_blob->mutable_cpu_data() + input_blob->offset(n);
      Dtype* eq_filter_data = eq_filter->mutable_cpu_data() + eq_filter->offset(n);
      int len = input_blob->channels() * input_blob->height() * input_blob->width();
      Dtype predicted_val = caffe_cpu_dot<Dtype>(len, input_data, eq_filter_data);

      error += (output_val - predicted_val) * (output_val - predicted_val);
      //LOG(INFO)<<"OUTTEST :"<<output_val << " / "<<predicted_val;
    }
    return error;
  }

  template <typename Dtype>
  const vector<Blob<Dtype>*>& FeedbackNet<Dtype>::FeedbackForward(const vector<Blob<Dtype>*> & bottom, 
								  Dtype* loss) {
    // Copy bottom to internal bottom
    for (int i = 0; i < bottom.size(); ++i) {
      this->net_input_blobs_[i]->CopyFrom(*bottom[i]);
    }
    return FeedbackForwardPrefilled(loss);
  }


  template<typename Dtype>
  const vector<Blob<Dtype>*>& FeedbackNet<Dtype>::FeedbackForwardPrefilled(string startLayer, int channel, int offset, int max_iterations) {
    int startLayerIdx = this->layer_names_index_[startLayer];
    FeedbackForwardPrefilled(NULL, startLayerIdx, channel, offset, max_iterations);
  }

  template<typename Dtype>
  const vector<Blob<Dtype>*>& FeedbackNet<Dtype>::FeedbackForwardPrefilled(Dtype* loss, int startLayerIdx, int channel, int offset, int max_iterations){
    this->forwardCompleteFlag = true;
    //reset layers
    for (int i = 0; i < this->layers_.size(); ++i) {
      this->layers_[i]->Reset();
    }

    if (loss != NULL) {
      *loss = Dtype(0.);
    }
    for (int i = 0; i < this->layers_.size(); ++i) {
      Dtype layer_loss = this->layers_[i]->Forward(this->bottom_vecs_[i], &(this->top_vecs_[i]));
      if (loss != NULL) {
	*loss += layer_loss;
      }
    }

    if(startLayerIdx == -1){
      //by default, don't use the last layer as feedback target
      //Since in most cases, the last layer is either prob, accuracy, or error layers
      //Note: layers_.size() - 1 is the last layer
      startLayerIdx = this->layers_.size()-2;
    }

    int k = 1;
    Blob<Dtype>* input_blob = (this->blobs_[0]).get();
    vector<int*> k_channels;
    vector<int*> k_offsets;
    for(int i = 0; i < k; ++i){
      int* channel_offset = new int[input_blob->num()];
      k_channels.push_back(channel_offset);
      int* in_channel_offset = new int[input_blob->num()];
      k_offsets.push_back(in_channel_offset);
    }
    
    SearchTopKNeurons(startLayerIdx, k, k_channels, k_offsets);
    int* channels = new int[input_blob->num()];
    int* offsets = new int[input_blob->num()];
    if(channel == -1 && offset == -1) {
      //get the largest output category
      channels = k_channels[0];
      offsets = k_offsets[0];
    }    
    else{
      //Using the selected neuron
      //LOG(INFO)<<channel << ":"<<offset<<" * "<<input_blob->num();
      for(int n = 0; n<input_blob->num(); ++n){
	channels[n] = channel;
	offsets[n] = offset;
      }
    }
    Dtype output_sum = (Dtype) 0.;
    Blob<Dtype>* top_output_blob = (this->top_vecs_[startLayerIdx])[0];
    Dtype* output_value = new Dtype[top_output_blob->num()];

    /*
     * For debug: output the ranking information
     * including chosen neurons, 
     * top k neurons in each iterations, etc.
     */

    /*
    LOG(INFO)<<"Initial Output Values:";
    for(int n = 0; n<top_output_blob->num(); ++n){
      output_value[n] = *(top_output_blob->cpu_data() 
			  + top_output_blob->offset(n, channels[n]) + offsets[n]);
      output_sum += output_value[n];
      LOG(INFO)<<"Using "<<channels[n] << " / "<<offsets[n]
	       <<" as target neurons for image "<<n
	       <<" : "<<output_value[n] << "\n";
      LOG(INFO)<<"Top "<<k<<" neurons for image "<<n<<":";      
      for(int i = 0; i<k; i++){
          Dtype value = *(top_output_blob->cpu_data()
			  + top_output_blob->offset(n, (k_channels[i])[n]) 
			  + (k_offsets[i])[n]);
          LOG(INFO)<<"[Channel: " <<(k_channels[i])[n] <<":"<< value << "] ";
      }
    }
    */

    int iteration  = 0;
    while(true){
      if (loss != NULL) {
	*loss = Dtype(0.);
      }
      UpdateEqFilter(startLayerIdx, channels, offsets);
      //Forward - don't deal with data layer
      for (int i = 1; i < this->layers_.size(); ++i) {
	Dtype layer_loss 
	  = this->layers_[i]->Forward(this->bottom_vecs_[i], &(this->top_vecs_[i]));
	if (loss != NULL) {
	  *loss += layer_loss;
	}
      }
      //LOG(INFO)<<"Feedback Iteration"<<iteration;
      Dtype new_output_sum = (Dtype) 0.;
      for(int n = 0; n<top_output_blob->num(); ++n){
	output_value[n] = *(top_output_blob->cpu_data() 
			    + top_output_blob->offset(n, channels[n]) + offsets[n]);
	new_output_sum += output_value[n];
	//LOG(INFO)<<output_value[n];
      }
      iteration ++;
      Dtype converge = (output_sum - new_output_sum) * (output_sum - new_output_sum);
      if(converge <= 1 || iteration >= max_iterations){
	//Doesnot allow too many iterations
	//LOG(INFO)<<"Converge in " <<iteration << " iterations!";
	break;
      }
      else{
	output_sum = new_output_sum;
      }
    }
    return this->net_output_blobs_;
  }

  INSTANTIATE_CLASS(FeedbackNet);

} //namespace caffe
