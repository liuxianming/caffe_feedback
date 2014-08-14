// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

using std::max;

namespace caffe {

  template <typename Dtype>
  void BAccuracyLayer<Dtype>::SetUp(
				   const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
    CHECK_EQ(bottom.size(), 2) << "Accuracy Layer takes two blobs as input.";
    CHECK_EQ(top->size(), 1) << "Accuracy Layer takes 1 output.";
    CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
    CHECK_EQ(bottom[0]->channels(), 1)
      << "The output must be single value";
    CHECK_EQ(bottom[1]->channels(), 1);
    CHECK_EQ(bottom[1]->height(), 1);
    CHECK_EQ(bottom[1]->width(), 1);
    (*top)[0]->Reshape(1, 2, 1, 1);
  }

  template <typename Dtype>
  Dtype BAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
					  vector<Blob<Dtype>*>* top) {
    Dtype accuracy = 0;
    Dtype logprob = 0;
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* bottom_label = bottom[1]->cpu_data();
    int num = bottom[0]->num();
    int dim = bottom[0]->count() / bottom[0]->num();
    for (int i = 0; i < num; ++i) {
      // Accuracy
      int class_label = ((bottom_data[i] > 0) ? 1 : 0);
      if (class_label == static_cast<int>(bottom_label[i])) {
	++accuracy;
      }
      Dtype prob = Dtype(1.0) / (exp( Dtype(-1.0) * bottom_data[i]) + 1);
      //Dtype prob = max(bottom_data[i * dim + static_cast<int>(bottom_label[i])],
      //		       Dtype(kLOG_THRESHOLD));
      logprob -= log(prob) * bottom_label[i];
    }
    // LOG(INFO) << "Accuracy: " << accuracy;
    (*top)[0]->mutable_cpu_data()[0] = accuracy / num;
    (*top)[0]->mutable_cpu_data()[1] = logprob / num;
    // Accuracy layer should not be used as a loss function.
    return Dtype(0);
  }

  INSTANTIATE_CLASS(BAccuracyLayer);

}  // namespace caffe
