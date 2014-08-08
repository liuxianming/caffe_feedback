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
class ConvolutionLayerTest : public ::testing::Test {
 protected:
  ConvolutionLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    blob_bottom_->Reshape(2, 3, 6, 4);
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~ConvolutionLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(ConvolutionLayerTest, Dtypes);

TYPED_TEST(ConvolutionLayerTest, TestSetup) {
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(4);
  shared_ptr<Layer<TypeParam> > layer(
      new ConvolutionLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 4);
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 1);
  // setting group should not change the shape
  convolution_param->set_num_output(3);
  convolution_param->set_group(3);
  layer.reset(new ConvolutionLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(ConvolutionLayerTest, TestCPUSimpleConvolution) {
  // We will simply see if the convolution layer carries out averaging well.
  FillerParameter filler_param;
  filler_param.set_value(1.);
  ConstantFiller<TypeParam> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(4);
  convolution_param->mutable_weight_filler()->set_type("constant");
  convolution_param->mutable_weight_filler()->set_value(1);
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<TypeParam> > layer(
      new ConvolutionLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  Caffe::set_mode(Caffe::CPU);
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // After the convolution, the output should all have output values 27.1
  const TypeParam* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], 27.1, 1e-4);
  }
}

TYPED_TEST(ConvolutionLayerTest, TestGPUSimpleConvolution) {
  // We will simply see if the convolution layer carries out averaging well.
  FillerParameter filler_param;
  filler_param.set_value(1.);
  ConstantFiller<TypeParam> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(4);
  convolution_param->mutable_weight_filler()->set_type("constant");
  convolution_param->mutable_weight_filler()->set_value(1);
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<TypeParam> > layer(
      new ConvolutionLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  Caffe::set_mode(Caffe::GPU);
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // After the convolution, the output should all have output values 27.1
  const TypeParam* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], 27.1, 1e-4);
  }
}

TYPED_TEST(ConvolutionLayerTest, TestCPUSimpleConvolutionGroup) {
  // We will simply see if the convolution layer carries out averaging well.
  FillerParameter filler_param;
  filler_param.set_value(1.);
  ConstantFiller<TypeParam> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  TypeParam* bottom_data = this->blob_bottom_->mutable_cpu_data();
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          bottom_data[this->blob_bottom_->offset(n, c, h, w)] = c;
        }
      }
    }
  }
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(3);
  convolution_param->set_group(3);
  convolution_param->mutable_weight_filler()->set_type("constant");
  convolution_param->mutable_weight_filler()->set_value(1);
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<TypeParam> > layer(
      new ConvolutionLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  Caffe::set_mode(Caffe::CPU);
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // After the convolution, the output should all have output values 9.1
  const TypeParam* top_data = this->blob_top_->cpu_data();
  for (int n = 0; n < this->blob_top_->num(); ++n) {
    for (int c = 0; c < this->blob_top_->channels(); ++c) {
      for (int h = 0; h < this->blob_top_->height(); ++h) {
        for (int w = 0; w < this->blob_top_->width(); ++w) {
          TypeParam data = top_data[this->blob_top_->offset(n, c, h, w)];
          EXPECT_NEAR(data, c * 9 + 0.1, 1e-4);
        }
      }
    }
  }
}


TYPED_TEST(ConvolutionLayerTest, TestGPUSimpleConvolutionGroup) {
  // We will simply see if the convolution layer carries out averaging well.
  FillerParameter filler_param;
  filler_param.set_value(1.);
  ConstantFiller<TypeParam> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  TypeParam* bottom_data = this->blob_bottom_->mutable_cpu_data();
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          bottom_data[this->blob_bottom_->offset(n, c, h, w)] = c;
        }
      }
    }
  }
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(3);
  convolution_param->set_group(3);
  convolution_param->mutable_weight_filler()->set_type("constant");
  convolution_param->mutable_weight_filler()->set_value(1);
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<TypeParam> > layer(
      new ConvolutionLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  Caffe::set_mode(Caffe::GPU);
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // After the convolution, the output should all have output values 9.1
  const TypeParam* top_data = this->blob_top_->cpu_data();
  for (int n = 0; n < this->blob_top_->num(); ++n) {
    for (int c = 0; c < this->blob_top_->channels(); ++c) {
      for (int h = 0; h < this->blob_top_->height(); ++h) {
        for (int w = 0; w < this->blob_top_->width(); ++w) {
          TypeParam data = top_data[this->blob_top_->offset(n, c, h, w)];
          EXPECT_NEAR(data, c * 9 + 0.1, 1e-4);
        }
      }
    }
  }
}


TYPED_TEST(ConvolutionLayerTest, TestCPUGradient) {
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(2);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  Caffe::set_mode(Caffe::CPU);
  ConvolutionLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

TYPED_TEST(ConvolutionLayerTest, TestCPUGradientGroup) {
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(3);
  convolution_param->set_group(3);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  Caffe::set_mode(Caffe::CPU);
  ConvolutionLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

TYPED_TEST(ConvolutionLayerTest, TestGPUGradient) {
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(2);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  Caffe::set_mode(Caffe::GPU);
  ConvolutionLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

TYPED_TEST(ConvolutionLayerTest, TestGPUGradientGroup) {
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(3);
  convolution_param->set_group(3);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  Caffe::set_mode(Caffe::GPU);
  ConvolutionLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

TYPED_TEST(ConvolutionLayerTest, TestCPUUpdateEqFilter) {
  // We will simply see if the convolution layer carries out averaging well.
  FillerParameter filler_param;
  filler_param.set_value(1.);
  ConstantFiller<TypeParam> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_pad(1);
  convolution_param->set_num_output(4);
  convolution_param->mutable_weight_filler()->set_type("uniform");
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.0);
  shared_ptr<Layer<TypeParam> > layer(
      new ConvolutionLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  Caffe::set_mode(Caffe::CPU);
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // After the convolution, the output should all have output values 27.1
  const TypeParam* top_data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
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
  
  //Compute EqFilter
  Blob<TypeParam>* const top_filter = new Blob<TypeParam>(2, 1, 1, 4*3*2);
  // fill the value for top_filter to test UpdateEqFilter
  // FillerParameter filler_param;
  filler_param.set_value(1.);
  // ConstantFiller<TypeParam> filler(filler_param);
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

TYPED_TEST(ConvolutionLayerTest, TestGPUUpdateEqFilter) {
  // We will simply see if the convolution layer carries out averaging well.
  FillerParameter filler_param;
  filler_param.set_value(1.);
  ConstantFiller<TypeParam> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_pad(1);
  convolution_param->set_num_output(4);
  convolution_param->mutable_weight_filler()->set_type("uniform");
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.0);
  shared_ptr<Layer<TypeParam> > layer(
      new ConvolutionLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  Caffe::set_mode(Caffe::GPU);
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // After the convolution, the output should all have output values 27.1
  const TypeParam* top_data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
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
  
  //Compute EqFilter
  Blob<TypeParam>* const top_filter = new Blob<TypeParam>(2, 1, 1, 4*3*2);
  // fill the value for top_filter to test UpdateEqFilter
  // FillerParameter filler_param;
  filler_param.set_value(1.);
  // ConstantFiller<TypeParam> filler(filler_param);
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

TYPED_TEST(ConvolutionLayerTest, TestCPUUpdateEqFilterGroup) {
  // We will simply see if the convolution layer carries out averaging well.
  FillerParameter filler_param;
  filler_param.set_value(1.);
  ConstantFiller<TypeParam> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  TypeParam* bottom_data = this->blob_bottom_->mutable_cpu_data();
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          bottom_data[this->blob_bottom_->offset(n, c, h, w)] = c;
        }
      }
    }
  }
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_pad(1);
  convolution_param->set_num_output(3);
  convolution_param->set_group(3);
  convolution_param->mutable_weight_filler()->set_type("uniform");
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0);
  shared_ptr<Layer<TypeParam> > layer(
      new ConvolutionLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  Caffe::set_mode(Caffe::CPU);
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // After the convolution, the output should all have output values 9.1
  const TypeParam* top_data = this->blob_top_->cpu_data();
  for (int n = 0; n < this->blob_top_->num(); ++n) {
    for (int c = 0; c < this->blob_top_->channels(); ++c) {
      for (int h = 0; h < this->blob_top_->height(); ++h) {
        for (int w = 0; w < this->blob_top_->width(); ++w) {
          TypeParam data = top_data[this->blob_top_->offset(n, c, h, w)];
          EXPECT_GE(data, 0);
        }
      }
    }
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
  Blob<TypeParam>* const top_filter = new Blob<TypeParam>(2, 1, 1, 4*3*2);
  // fill the value for top_filter to test UpdateEqFilter
  // FillerParameter filler_param;
  filler_param.set_value(1.);
  // ConstantFiller<TypeParam> filler(filler_param);
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

TYPED_TEST(ConvolutionLayerTest, TestGPUUpdateEqFilterGroup) {
  // We will simply see if the convolution layer carries out averaging well.
  FillerParameter filler_param;
  filler_param.set_value(1.);
  ConstantFiller<TypeParam> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  TypeParam* bottom_data = this->blob_bottom_->mutable_cpu_data();
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          bottom_data[this->blob_bottom_->offset(n, c, h, w)] = c;
        }
      }
    }
  }
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_pad(1);
  convolution_param->set_num_output(3);
  convolution_param->set_group(3);
  convolution_param->mutable_weight_filler()->set_type("uniform");
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0);
  shared_ptr<Layer<TypeParam> > layer(
      new ConvolutionLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  Caffe::set_mode(Caffe::GPU);
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // After the convolution, the output should all have output values 9.1
  const TypeParam* top_data = this->blob_top_->cpu_data();
  for (int n = 0; n < this->blob_top_->num(); ++n) {
    for (int c = 0; c < this->blob_top_->channels(); ++c) {
      for (int h = 0; h < this->blob_top_->height(); ++h) {
        for (int w = 0; w < this->blob_top_->width(); ++w) {
          TypeParam data = top_data[this->blob_top_->offset(n, c, h, w)];
          EXPECT_GE(data, 0);
        }
      }
    }
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
  Blob<TypeParam>* const top_filter = new Blob<TypeParam>(2, 1, 1, 4*3*2);
  // fill the value for top_filter to test UpdateEqFilter
  // FillerParameter filler_param;
  filler_param.set_value(1.);
  // ConstantFiller<TypeParam> filler(filler_param);
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
