// Copyright 2014 BVLC and contributors.

#include <cstring>
#include <vector>

#include "cuda_runtime.h"
#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;

template <typename Dtype>
class PoolingLayerTest : public ::testing::Test {
 protected:
  PoolingLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    blob_bottom_->Reshape(2, 3, 6, 5);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~PoolingLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(PoolingLayerTest, Dtypes);

TYPED_TEST(PoolingLayerTest, TestSetup) {
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_size(3);
  pooling_param->set_stride(2);
  PoolingLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 2);
}

TYPED_TEST(PoolingLayerTest, TestSetupPadded) {
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_size(3);
  pooling_param->set_stride(2);
  pooling_param->set_pad(1);
  pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
  PoolingLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 3);
}

/*
TYPED_TEST(PoolingLayerTest, PrintGPUBackward) {
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_size(3);
  pooling_param->set_stride(2);
  pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
  Caffe::set_mode(Caffe::GPU);
  PoolingLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    cout << "bottom data " << i << " " << this->blob_bottom_->cpu_data()[i] << endl;
  }
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    cout << "top data " << i << " " << this->blob_top_->cpu_data()[i] << endl;
  }

  for (int i = 0; i < this->blob_top_->count(); ++i) {
    this->blob_top_->mutable_cpu_diff()[i] = 1.;
  }
  layer.Backward(this->blob_top_vec_, true, &(this->blob_bottom_vec_));
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    cout << "bottom diff " << i << " " << this->blob_bottom_->cpu_diff()[i] << endl;
  }
}
*/

TYPED_TEST(PoolingLayerTest, TestCPUGradientMax) {
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_size(3);
  pooling_param->set_stride(2);
  pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
  Caffe::set_mode(Caffe::CPU);
  PoolingLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-4, 1e-2);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

TYPED_TEST(PoolingLayerTest, TestGPUGradientMax) {
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_size(3);
  pooling_param->set_stride(2);
  pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
  Caffe::set_mode(Caffe::GPU);
  PoolingLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-4, 1e-2);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}


TYPED_TEST(PoolingLayerTest, TestCPUForwardAve) {
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_size(3);
  pooling_param->set_stride(1);
  pooling_param->set_pad(1);
  pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
  Caffe::set_mode(Caffe::CPU);
  this->blob_bottom_->Reshape(1, 1, 3, 3);
  FillerParameter filler_param;
  filler_param.set_value(TypeParam(2));
  ConstantFiller<TypeParam> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  PoolingLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 3);
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  TypeParam epsilon = 1e-5;
  EXPECT_NEAR(this->blob_top_->cpu_data()[0], 8.0 / 9, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[1], 4.0 / 3, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[2], 8.0 / 9, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[3], 4.0 / 3, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[4], 2.0    , epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[5], 4.0 / 3, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[6], 8.0 / 9, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[7], 4.0 / 3, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[8], 8.0 / 9, epsilon);
}


TYPED_TEST(PoolingLayerTest, TestGPUForwardAve) {
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_size(3);
  pooling_param->set_stride(1);
  pooling_param->set_pad(1);
  pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
  Caffe::set_mode(Caffe::GPU);
  this->blob_bottom_->Reshape(1, 1, 3, 3);
  FillerParameter filler_param;
  filler_param.set_value(TypeParam(2));
  ConstantFiller<TypeParam> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  PoolingLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 3);
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  TypeParam epsilon = 1e-5;
  EXPECT_NEAR(this->blob_top_->cpu_data()[0], 8.0 / 9, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[1], 4.0 / 3, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[2], 8.0 / 9, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[3], 4.0 / 3, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[4], 2.0    , epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[5], 4.0 / 3, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[6], 8.0 / 9, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[7], 4.0 / 3, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[8], 8.0 / 9, epsilon);
}


TYPED_TEST(PoolingLayerTest, TestCPUGradientAve) {
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_size(3);
  pooling_param->set_stride(2);
  pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
  Caffe::set_mode(Caffe::CPU);
  PoolingLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}


TYPED_TEST(PoolingLayerTest, TestGPUGradientAve) {
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_size(3);
  pooling_param->set_stride(2);
  pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
  Caffe::set_mode(Caffe::GPU);
  PoolingLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}


TYPED_TEST(PoolingLayerTest, TestCPUGradientAvePadded) {
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_size(3);
  pooling_param->set_stride(2);
  pooling_param->set_pad(2);
  pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
  Caffe::set_mode(Caffe::CPU);
  PoolingLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}


TYPED_TEST(PoolingLayerTest, TestGPUGradientAvePadded) {
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_size(3);
  pooling_param->set_stride(2);
  pooling_param->set_pad(2);
  pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
  Caffe::set_mode(Caffe::GPU);
  PoolingLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

TYPED_TEST(PoolingLayerTest, TestCPUUpdateEqFilter) {
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_size(3);
  pooling_param->set_stride(2);
  pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
  Caffe::set_mode(Caffe::CPU);
   
  // NOTE: Due to fully connected layer feedforward use only a fixed set of weights
  // no matter how many images are there. Hence, during testing, fully connected
  // layer cannot test multiple batches if there is pooling or relu layers because 
  // eq_filter won't be the same for each image. The first dimension of blob_bottom
  // must be set as 1 !
  // reshape bottom to test different scenarios
  this->blob_bottom_->Reshape(1, 3, 6, 5);
  FillerParameter filler_param;
  GaussianFiller<TypeParam> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  shared_ptr<Layer<TypeParam> > layer(
      new PoolingLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_bottom_->num(), 1);
  EXPECT_EQ(this->blob_bottom_->channels(), 3);
  EXPECT_EQ(this->blob_bottom_->height(), 6);
  EXPECT_EQ(this->blob_bottom_->width(), 5);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  // Another note: current pooling layer uses ceil(n+2*p-k)/s+1) as the output size
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 2);
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  TypeParam epsilon = 1e-5;
  const TypeParam* top_data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  Blob<TypeParam> * pooling_mask = reinterpret_cast<PoolingLayer<TypeParam>*>(layer.get())->GetMask(); 
  for (int i = 0; i < count; ++i) {
    EXPECT_GE(top_data[i], 0.);
  }

  // Feedforward second layer, a fully connected layer with all weights 1 and 
  // bias 0, sum up the data from first layer
  Blob<TypeParam>* const blob_top_fc = new Blob<TypeParam>();
  vector<Blob<TypeParam>*> blob_top_fc_vec;
  blob_top_fc_vec.push_back(blob_top_fc);
  
  LayerParameter layer_param_fc;
  InnerProductParameter* inner_product_param_fc = 
    layer_param_fc.mutable_inner_product_param();
  inner_product_param_fc->set_num_output(1);
  inner_product_param_fc->mutable_weight_filler()->set_type("constant");
  inner_product_param_fc->mutable_weight_filler()->set_value(1);
  inner_product_param_fc->mutable_bias_filler()->set_type("constant");
  inner_product_param_fc->mutable_bias_filler()->set_value(0);
  shared_ptr<InnerProductLayer<TypeParam> > layer_fc(
      new InnerProductLayer<TypeParam>(layer_param_fc));
  layer_fc->SetUp(this->blob_top_vec_, &(blob_top_fc_vec));
  layer_fc->Forward(this->blob_top_vec_, &(blob_top_fc_vec));
  const TypeParam* data_fc = blob_top_fc->cpu_data();
  const int count_fc = blob_top_fc->count();
  
  // Compute EqFilter
  Blob<TypeParam>* const top_filter = 
    new Blob<TypeParam>(this->blob_top_->num(), 1, 1,
        this->blob_top_->channels() * this->blob_top_->height() * this->blob_top_->width());
  // Fill the value for top_filter to test UpdateEqFilter
  FillerParameter filler_param_eq;
  filler_param_eq.set_value(1.);
  ConstantFiller<TypeParam> filler_eq(filler_param_eq);
  filler_eq.Fill(top_filter);
  layer->UpdateEqFilter(top_filter, this->blob_bottom_vec_);

  // Feedforward the computed eqfilter, final results expected to be 
  Blob<TypeParam>* const blob_top_eq = new Blob<TypeParam>();
  vector<Blob<TypeParam>*> blob_top_eq_vec; 
  blob_top_eq_vec.push_back(blob_top_eq);
  
  LayerParameter layer_param_eq;
  InnerProductParameter* inner_product_param_eq = 
    layer_param_eq.mutable_inner_product_param();
  inner_product_param_eq->set_num_output(1);
  inner_product_param_eq->set_bias_term(false);
  shared_ptr<InnerProductLayer<TypeParam> > layer_eq(
    new InnerProductLayer<TypeParam>(layer_param_eq));
  layer_eq->SetUp(this->blob_bottom_vec_, &(blob_top_eq_vec));
  layer_eq->blobs()[0]->CopyFrom(*layer->eq_filter(), false, true);
  layer_eq->Forward(this->blob_bottom_vec_, &(blob_top_eq_vec));
  const TypeParam* data_eq = blob_top_eq->cpu_data();
  const int count_eq = blob_top_eq->count();

  ASSERT_EQ(count_fc, count_eq);
  for (int i = 0; i < count_eq; ++i) {
    EXPECT_NEAR(data_fc[i], data_eq[i], 1e-4);
  }

  delete blob_top_fc; delete blob_top_eq; delete top_filter;
}

TYPED_TEST(PoolingLayerTest, TestGPUUpdateEqFilter) {
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_size(3);
  pooling_param->set_stride(2);
  pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
  Caffe::set_mode(Caffe::GPU);
   
  // NOTE: Due to fully connected layer feedforward use only a fixed set of weights
  // no matter how many images are there. Hence, during testing, fully connected
  // layer cannot test multiple batches if there is pooling or relu layers because 
  // eq_filter won't be the same for each image. The first dimension of blob_bottom
  // must be set as 1 !
  // reshape bottom to test different scenarios
  this->blob_bottom_->Reshape(1, 3, 6, 5);
  FillerParameter filler_param;
  GaussianFiller<TypeParam> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  shared_ptr<Layer<TypeParam> > layer(
      new PoolingLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_bottom_->num(), 1);
  EXPECT_EQ(this->blob_bottom_->channels(), 3);
  EXPECT_EQ(this->blob_bottom_->height(), 6);
  EXPECT_EQ(this->blob_bottom_->width(), 5);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  // Another note: current pooling layer uses ceil(n+2*p-k)/s+1) as the output size
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 2);
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  TypeParam epsilon = 1e-5;
  const TypeParam* top_data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  Blob<TypeParam> * pooling_mask = reinterpret_cast<PoolingLayer<TypeParam>*>(layer.get())->GetMask(); 
  for (int i = 0; i < count; ++i) {
    EXPECT_GE(top_data[i], 0.);
  }

  // Feedforward second layer, a fully connected layer with all weights 1 and 
  // bias 0, sum up the data from first layer
  Blob<TypeParam>* const blob_top_fc = new Blob<TypeParam>();
  vector<Blob<TypeParam>*> blob_top_fc_vec;
  blob_top_fc_vec.push_back(blob_top_fc);
  
  LayerParameter layer_param_fc;
  InnerProductParameter* inner_product_param_fc = 
    layer_param_fc.mutable_inner_product_param();
  inner_product_param_fc->set_num_output(1);
  inner_product_param_fc->mutable_weight_filler()->set_type("constant");
  inner_product_param_fc->mutable_weight_filler()->set_value(1);
  inner_product_param_fc->mutable_bias_filler()->set_type("constant");
  inner_product_param_fc->mutable_bias_filler()->set_value(0);
  shared_ptr<InnerProductLayer<TypeParam> > layer_fc(
      new InnerProductLayer<TypeParam>(layer_param_fc));
  layer_fc->SetUp(this->blob_top_vec_, &(blob_top_fc_vec));
  layer_fc->Forward(this->blob_top_vec_, &(blob_top_fc_vec));
  const TypeParam* data_fc = blob_top_fc->cpu_data();
  const int count_fc = blob_top_fc->count();
  
  // Compute EqFilter
  Blob<TypeParam>* const top_filter = 
    new Blob<TypeParam>(this->blob_top_->num(), 1, 1,
        this->blob_top_->channels() * this->blob_top_->height() * this->blob_top_->width());
  // Fill the value for top_filter to test UpdateEqFilter
  FillerParameter filler_param_eq;
  filler_param_eq.set_value(1.);
  ConstantFiller<TypeParam> filler_eq(filler_param_eq);
  filler_eq.Fill(top_filter);
  layer->UpdateEqFilter(top_filter, this->blob_bottom_vec_);

  // Feedforward the computed eqfilter, final results expected to be 
  Blob<TypeParam>* const blob_top_eq = new Blob<TypeParam>();
  vector<Blob<TypeParam>*> blob_top_eq_vec; 
  blob_top_eq_vec.push_back(blob_top_eq);
  
  LayerParameter layer_param_eq;
  InnerProductParameter* inner_product_param_eq = 
    layer_param_eq.mutable_inner_product_param();
  inner_product_param_eq->set_num_output(1);
  inner_product_param_eq->set_bias_term(false);
  shared_ptr<InnerProductLayer<TypeParam> > layer_eq(
    new InnerProductLayer<TypeParam>(layer_param_eq));
  layer_eq->SetUp(this->blob_bottom_vec_, &(blob_top_eq_vec));
  layer_eq->blobs()[0]->CopyFrom(*layer->eq_filter(), false, true);
  layer_eq->Forward(this->blob_bottom_vec_, &(blob_top_eq_vec));
  const TypeParam* data_eq = blob_top_eq->cpu_data();
  const int count_eq = blob_top_eq->count();

  ASSERT_EQ(count_fc, count_eq);
  for (int i = 0; i < count_eq; ++i) {
    EXPECT_NEAR(data_fc[i], data_eq[i], 1e-4);
  }

  delete blob_top_fc; delete blob_top_eq; delete top_filter;
}

}  // namespace caffe
