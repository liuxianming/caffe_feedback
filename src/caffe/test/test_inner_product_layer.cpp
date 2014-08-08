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
class InnerProductLayerTest : public ::testing::Test {
 protected:
  InnerProductLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~InnerProductLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

/*
template <typename Dtype>
class InnerProductLayerTest : public ::testing::Test {
 protected:
  InnerProductLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()),
        top_filter_(new Blob<Dtype>(2, 10, 1, 1)),
        blob_top_fc_(new Blob<Dtype>()),
        blob_top_eq_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);

    // fill the value for top_filter to test UpdateEqFilter
    filler_param.set_value(1.);
    ConstantFiller<Dtype> filler_c(filler_param);
    filler_c.Fill(this->top_filter_);
    blob_top_fc_vec_.push_back(blob_top_fc_);
    blob_top_eq_vec_.push_back(blob_top_eq_);
  }
  virtual ~InnerProductLayerTest() { delete blob_bottom_; delete blob_top_; 
    delete top_filter_; delete blob_top_fc_; delete blob_top_eq_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  
  // variables for testing UpdateEqFilter
  Blob<Dtype>* const top_filter_;
  Blob<Dtype>* const blob_top_fc_;
  vector<Blob<Dtype>*> blob_top_fc_vec_;
  Blob<Dtype>* const blob_top_eq_;
  vector<Blob<Dtype>*> blob_top_eq_vec_; 
};*/

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(InnerProductLayerTest, Dtypes);

TYPED_TEST(InnerProductLayerTest, TestSetUp) {
  LayerParameter layer_param;
  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  inner_product_param->set_num_output(10);
  shared_ptr<InnerProductLayer<TypeParam> > layer(
      new InnerProductLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 10);
}

TYPED_TEST(InnerProductLayerTest, TestCPU) {
  LayerParameter layer_param;
  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  Caffe::set_mode(Caffe::CPU);
  inner_product_param->set_num_output(10);
  inner_product_param->mutable_weight_filler()->set_type("uniform");
  inner_product_param->mutable_bias_filler()->set_type("uniform");
  inner_product_param->mutable_bias_filler()->set_min(1);
  inner_product_param->mutable_bias_filler()->set_max(2);
  shared_ptr<InnerProductLayer<TypeParam> > layer(
      new InnerProductLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  const TypeParam* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  for (int i = 0; i < count; ++i) {
    EXPECT_GE(data[i], 1.);
  }
}

TYPED_TEST(InnerProductLayerTest, TestGPU) {
  if (sizeof(TypeParam) == 4 || CAFFE_TEST_CUDA_PROP.major >= 2) {
    LayerParameter layer_param;
    InnerProductParameter* inner_product_param =
        layer_param.mutable_inner_product_param();
    Caffe::set_mode(Caffe::GPU);
    inner_product_param->set_num_output(10);
    inner_product_param->mutable_weight_filler()->set_type("uniform");
    inner_product_param->mutable_bias_filler()->set_type("uniform");
    inner_product_param->mutable_bias_filler()->set_min(1);
    inner_product_param->mutable_bias_filler()->set_max(2);
    shared_ptr<InnerProductLayer<TypeParam> > layer(
      new InnerProductLayer<TypeParam>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
    layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
    const TypeParam* data = this->blob_top_->cpu_data();
    const int count = this->blob_top_->count();
    for (int i = 0; i < count; ++i) {
      EXPECT_GE(data[i], 1.);
    }
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(InnerProductLayerTest, TestCPUGradient) {
  LayerParameter layer_param;
  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  Caffe::set_mode(Caffe::CPU);
  inner_product_param->set_num_output(10);
  inner_product_param->mutable_weight_filler()->set_type("gaussian");
  inner_product_param->mutable_bias_filler()->set_type("gaussian");
  inner_product_param->mutable_bias_filler()->set_min(1);
  inner_product_param->mutable_bias_filler()->set_max(2);
  InnerProductLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

TYPED_TEST(InnerProductLayerTest, TestGPUGradient) {
  if (sizeof(TypeParam) == 4 || CAFFE_TEST_CUDA_PROP.major >= 2) {
    LayerParameter layer_param;
    InnerProductParameter* inner_product_param =
        layer_param.mutable_inner_product_param();
    Caffe::set_mode(Caffe::GPU);
    inner_product_param->set_num_output(10);
    inner_product_param->mutable_weight_filler()->set_type("gaussian");
    inner_product_param->mutable_bias_filler()->set_type("gaussian");
    InnerProductLayer<TypeParam> layer(layer_param);
    GradientChecker<TypeParam> checker(1e-2, 1e-2);
    checker.CheckGradient(&layer, &(this->blob_bottom_vec_),
        &(this->blob_top_vec_));
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(InnerProductLayerTest, TestCPUUpdateEqFilter) {
  LayerParameter layer_param;
  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  Caffe::set_mode(Caffe::CPU);
  inner_product_param->set_num_output(10);
  inner_product_param->mutable_weight_filler()->set_type("uniform");
  inner_product_param->mutable_bias_filler()->set_type("constant");
  inner_product_param->mutable_bias_filler()->set_value(0);
  //inner_product_param->mutable_bias_filler()->set_type("uniform");
  //inner_product_param->mutable_bias_filler()->set_min(1);
  //inner_product_param->mutable_bias_filler()->set_max(2);
  shared_ptr<InnerProductLayer<TypeParam> > layer(
      new InnerProductLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  const TypeParam* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  for (int i = 0; i < count; ++i) {
    LOG(ERROR) << data[i];
    EXPECT_GE(data[i], 0.);
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
  
  //Compute EqFilter
  Blob<TypeParam>* const top_filter = new Blob<TypeParam>(2, 1, 1, 10);
  // fill the value for top_filter to test UpdateEqFilter
  FillerParameter filler_param;
  filler_param.set_value(1.);
  ConstantFiller<TypeParam> filler(filler_param);
  filler.Fill(top_filter);
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

TYPED_TEST(InnerProductLayerTest, TestGPUUpdateEqFilter) {
  LayerParameter layer_param;
  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  Caffe::set_mode(Caffe::GPU);
  inner_product_param->set_num_output(10);
  inner_product_param->mutable_weight_filler()->set_type("uniform");
  inner_product_param->mutable_bias_filler()->set_type("constant");
  inner_product_param->mutable_bias_filler()->set_value(0);
  //inner_product_param->mutable_bias_filler()->set_type("uniform");
  //inner_product_param->mutable_bias_filler()->set_min(1);
  //inner_product_param->mutable_bias_filler()->set_max(2);
  shared_ptr<InnerProductLayer<TypeParam> > layer(
      new InnerProductLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  const TypeParam* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  for (int i = 0; i < count; ++i) {
    LOG(ERROR) << data[i];
    EXPECT_GE(data[i], 0.);
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
  
  //Compute EqFilter
  Blob<TypeParam>* const top_filter = new Blob<TypeParam>(2, 1, 1, 10);
  // fill the value for top_filter to test UpdateEqFilter
  FillerParameter filler_param;
  filler_param.set_value(1.);
  ConstantFiller<TypeParam> filler(filler_param);
  filler.Fill(top_filter);
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
