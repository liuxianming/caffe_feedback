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

  //Forward process using FeedbackNet
  const vector<Blob<float>*>& result = caffe_test_net.ForwardPrefilled();
  Blob<float>* input_img = (caffe_test_net.blobs()[0]).get();

  //Read image mean from protobuffer file (indicated by argv[3])
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(argv[3], &blob_proto);
  Blob<float> data_mean;
  data_mean.FromProto(blob_proto);

  bool test_flag = true;

  //start feedback
  caffe_test_net.Visualize("conv2", 16, 50, test_flag);

  if(test_flag == false){
      Blob<float>* visualization = caffe_test_net.GetVisualization();

      float* imagedata = new float[data_mean.count()];

      for(int n = 0; n<visualization->num(); n++) {
          //visualize the i-th image
          float* blobdata = visualization->mutable_cpu_data() + visualization->offset(n);
          for (int i=0; i<data_mean.count(); i++){
              //          *(imagedata+i) = *(blobdata + i) + *(data_mean.mutable_cpu_data() + i);
              *(imagedata+i) = *(blobdata + i) ;
          }
          std::ostringstream convert;
          convert << n <<".jpg";
          string filename = convert.str();
          LOG(INFO)<<"Writing data to image "<<filename<<" ...";
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
  LOG(INFO)<<"Done";
}
