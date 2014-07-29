//Copyright: Xianming Liu, July 2014
//Used to test the function of image output from data blobs

#include <cuda_runtime.h>

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
  if (argc < 3 || argc > 5){
      LOG(ERROR)<<"test_data_to_image net_proto pretrained_net_proto [data_mean_proto]";
      return 1;
  }

  LOG(ERROR) << "Using CPU";
  Caffe::set_mode(Caffe::CPU);

  FeedbackNet<float> caffe_test_net(argv[1]);
  caffe_test_net.CopyTrainedLayersFrom(argv[2]);

  for(int iter = 0; iter < atoi(argv[3]); ++iter){
      //Forward process using FeedbackNet
      LOG(INFO)<<"Processing image feedforward...";
      const vector<Blob<float>*>& result = caffe_test_net.ForwardPrefilled();
      LOG(INFO)<<"Feedforward complete!";
      Blob<float>* input_img = (caffe_test_net.blobs()[0]).get();

      bool test_flag = false;
      //start feedback
      //caffe_test_net.Visualize("conv3", 5, 3, 3, test_flag);
      LOG(INFO)<<"Start visualization";
      caffe_test_net.VisualizeTopKNeurons("fc8", 1, true);

      if(test_flag == false){
          Blob<float>* visualization = caffe_test_net.GetVisualization();

          float* imagedata = new float[visualization->count()];

          for(int n = 0; n<visualization->num(); n++) {
              //visualize the i-th image
              float* blobdata = visualization->mutable_cpu_data() + visualization->offset(n);
              for (int i=0; i<visualization->count(); i++){
                  *(imagedata+i) = *(blobdata + i) ;
              }
              std::ostringstream convert;
              convert << iter <<"_"<<n <<".jpg";
              string filename = convert.str();
              caffe::WriteDataToImage<float>(filename,
                  visualization->channels(),
                  visualization->height(),
                  visualization->width(),
                  imagedata
              );
          }
          //clear
          delete [] imagedata;
      }
  }
  LOG(INFO)<<"Done";
}
