//Copyright: Xianming Liu, July 2014
//Used to test the function of image output from data blobs

#include <cuda_runtime.h>
#include <time.h>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <sstream>
#include <string>

#include "caffe/caffe.hpp"

using namespace caffe;

int main(int argc, char** argv){

  //::google::InitGoogleLogging(argv[0]);

  Caffe::set_phase(Caffe::TEST);
  if (argc < 3 || argc > 6){
      LOG(ERROR)<<"test_data_to_image net_proto pretrained_net_proto number_of_iterations starting_layer [Top K]";
      return 1;
  }

  LOG(ERROR) << "Using CPU";
  Caffe::set_mode(Caffe::GPU);

  FeedbackNet<float> caffe_test_net(argv[1]);
  caffe_test_net.CopyTrainedLayersFrom(argv[2]);

  for(int iter = 0; iter < atoi(argv[3]); ++iter){
      bool test_flag = false;
      //start feedback
      string startLayer(argv[4]);
      int topK = 1;
      if(argc == 6){
	topK = atoi(argv[5]);
      }
      LOG(INFO)<<"Start visualization";
      if(topK > 0){      
	//Forward process using FeedbackNet
	LOG(INFO)<<"Processing image feedforward...";
	const vector<Blob<float>*>& result = caffe_test_net.FeedbackForwardPrefilled("fc8");
	//const vector<Blob<float>*>& result = caffe_test_net.ForwardPrefilled();
	LOG(INFO)<<"Feedforward complete!";
	caffe_test_net.VisualizeTopKNeurons(startLayer, topK, false);
      }
      else{
	srand(time(NULL));
	int rand_idx = rand() % 1000;
	LOG(INFO)<<"Using Random Neurons: "<<rand_idx;
	//Forward process using FeedbackNet
	LOG(INFO)<<"Processing image feedforward...";
	const vector<Blob<float>*>& result 
	  = caffe_test_net.FeedbackForwardPrefilled("fc8", rand_idx, 0);
	LOG(INFO)<<"Feedforward complete!";
	LOG(INFO)<<"Visualizing using "<<rand_idx<<"-th neuron";
	caffe_test_net.Visualize(startLayer, rand_idx, 0, 0, test_flag);
      }
      if(test_flag == false){
          Blob<float>* visualization = caffe_test_net.GetVisualization();
	  std::ostringstream convert;
	  convert << iter <<"_";
	  string prefix = convert.str();
	  caffe_test_net.DrawVisualization("./", prefix);
      }
  }
  LOG(INFO)<<"Done";
}
