//Copyright: Xianming Liu, July 2014
//Used to test the function of image output from data blobs

#include <cuda_runtime.h>

#include <cstring>
#include <cstdlib>
#include <vector>

#include "caffe/caffe.hpp"

using namespace caffe;

int main(int argc, char** argv){
  Caffe::set_phase(Caffe::TEST);
  if (argc != 3){
      LOG(ERROR)<<"test_data_to_image net_proto pretrained_net_proto";
      return 1;
  }

  LOG(ERROR) << "Using CPU";
  Caffe::set_mode(Caffe::CPU);

  FeedbackNet<float> caffe_test_net(argv[1]);
  caffe_test_net.CopyTrainedLayersFrom(argv[2]);
}
