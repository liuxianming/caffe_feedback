// Copyright 2014 BVLC and contributors.
//
// This script converts the CIFAR dataset to the leveldb format used
// by caffe to perform classification.
// Usage:
//    convert_cifar_data input_folder output_db_file
// The CIFAR dataset could be downloaded at
//    http://www.cs.toronto.edu/~kriz/cifar.html

// In this version, a noisy cifar2 dataset will be generated
// By Xianming Liu

#include <google/protobuf/text_format.h>
#include <glog/logging.h>
#include <leveldb/db.h>

#include <stdint.h>
#include <fstream>  // NOLINT(readability/streams)
#include <string>

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

#include "caffe/common.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/proto/caffe.pb.h"

using std::string;

const int kCIFARSize = 32;
const int kCIFARImageNBytes = 3072;
const int kCIFARBatchSize = 10000;
const int kCIFARTrainBatches = 5;

void read_image(std::ifstream *file, int *label, char *buffer) {
    char label_char;
    file->read(&label_char, 1);
    *label = label_char;
    file->read(buffer, kCIFARImageNBytes);
    return;
}

void add_noise(char* buffer, int level) {
  //Add noise to image data, with length kCIFARImageNBytes
  float mean = 0.0;
  float sigma = 1.0;

  //Using boost::random to generate Gaussian noise, set mean = 0 and sigma = 1
  boost::normal_distribution<float> random_distribution(mean, sigma);
  boost::variate_generator<caffe::rng_t*,  boost::normal_distribution<float> >
    variate_generator(caffe::caffe_rng(), random_distribution);
  for (int i = 0; i < kCIFARImageNBytes ; ++i) {
    float noise = variate_generator() * level;
    float p_val = (float)(buffer[i]);
    p_val = p_val + noise;
    if (p_val > 255.0) p_val = 255.0;
    if (p_val < 0) p_val = 0.0;

    //turn the p_val into char format
    buffer[i] = (unsigned char) p_val;
  }
}

void convert_dataset(const string &input_folder, const string &output_folder, int level) {
    // Leveldb options
    leveldb::Options options;
    options.create_if_missing = true;
    options.error_if_exists = true;
    // Data buffer
    int label;
    char str_buffer[kCIFARImageNBytes];
    string value;
    caffe::Datum datum;
    datum.set_channels(3);
    datum.set_height(kCIFARSize);
    datum.set_width(kCIFARSize);

    LOG(INFO) << "Writing Training data";
    leveldb::DB *train_db;
    leveldb::Status status;
    status = leveldb::DB::Open(options, output_folder + "/cifar-train-leveldb",
                               &train_db);
    CHECK(status.ok()) << "Failed to open leveldb.";

    int counter = 0;
    
    // count number of positives and negatives for training and testing
    int num_train[2] = {0};
    int num_test[2] = {0};

    for (int fileid = 0; fileid < kCIFARTrainBatches; ++fileid) {
        // Open files
        LOG(INFO) << "Training Batch " << fileid + 1;
        snprintf(str_buffer, kCIFARImageNBytes, "/data_batch_%d.bin", fileid + 1);
        std::ifstream data_file((input_folder + str_buffer).c_str(),
                                std::ios::in | std::ios::binary);
        CHECK(data_file) << "Unable to open train file #" << fileid + 1;

        for (int itemid = 0; itemid < kCIFARBatchSize; ++itemid) {
            read_image(&data_file, &label, str_buffer);

            //Create a dataset containing "dog = 5" as positive and others as negative
	    //for other classes, using some counter to select image
            if (label == 5) {
		LOG(INFO)<<"Select image #" << itemid << " from batch #" << fileid << " into training. "
			 <<"Label changes "<<label << "->"<<1;
                label = 1;
                num_train[0] += 1;
            }
            else{
	      counter++;
	      //sample one image every 9 
	      if (counter % 9 == 1) {
		LOG(INFO)<<"Select image #" << itemid << " from batch #" << fileid << " into training. "
			 <<"Label changes "<<label << "->"<<0;
		label = 0;
		num_train[1] += 1;
	      }
	      else{
		continue;
	      }
	    }

	    //add noise to str_buffer
	    add_noise(str_buffer, level);

            datum.set_label(label);
            datum.set_data(str_buffer, kCIFARImageNBytes);
            datum.SerializeToString(&value);
            snprintf(str_buffer, kCIFARImageNBytes, "%05d",
                     fileid * kCIFARBatchSize + itemid);
            train_db->Put(leveldb::WriteOptions(), string(str_buffer), value);
        }
    }

    LOG(INFO) << "Writing Testing data";
    leveldb::DB *test_db;
    CHECK(leveldb::DB::Open(options, output_folder + "/cifar-test-leveldb",
                            &test_db).ok()) << "Failed to open leveldb.";
    // Open files
    std::ifstream data_file((input_folder + "/test_batch.bin").c_str(),
                            std::ios::in | std::ios::binary);
    CHECK(data_file) << "Unable to open test file.";

    //reset the counter
    counter = 0;
    for (int itemid = 0; itemid < kCIFARBatchSize; ++itemid) {
        read_image(&data_file, &label, str_buffer);
	if (label == 5) {
	    LOG(INFO)<<"Select image #" << itemid << " into testing. "
		     <<"Label changes "<<label << "->"<<1;
	  label = 1;
	  num_test[0] += 1;
	}
	else{
	  ++counter;
	  //sample one image every 9 
	  if (counter % 9 == 1) {
	    LOG(INFO)<<"Select image #" << itemid << " into testing. "
		     <<"Label changes "<<label << "->"<<0;
	    label = 0;
	    num_test[1] += 1;
	  }
	  else{
	    continue;
	  }
	}

	//add noise to str_buffer
	add_noise(str_buffer, level);

        datum.set_label(label);
        datum.set_data(str_buffer, kCIFARImageNBytes);
        datum.SerializeToString(&value);
        snprintf(str_buffer, kCIFARImageNBytes, "%05d", itemid);
        test_db->Put(leveldb::WriteOptions(), string(str_buffer), value);
    }

    LOG(INFO) << "Number of positive training images: " << num_train[0];
    LOG(INFO) << "Number of negative training images: " << num_train[1];
    LOG(INFO) << "Number of positive testing images: " << num_test[0];
    LOG(INFO) << "Number of negative testing images: " << num_test[1];
    delete train_db;
    delete test_db;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        printf("This script converts the CIFAR dataset to the leveldb format used\n"
               "by caffe to perform classification.\n"
	       "Modification:\n"
	       "\t Using two classes, and add noise to images"
               "Usage:\n"
               "    convert_cifar_data input_folder output_folder noise_level\n"
               "Where the input folder should contain the binary batch files.\n"
	       "noise_level is an integer to indicate the level of noise, e.g., 30.\n"
               "The CIFAR dataset could be downloaded at\n"
               "    http://www.cs.toronto.edu/~kriz/cifar.html\n"
               "You should gunzip them after downloading.\n");
    } else {
        google::InitGoogleLogging(argv[0]);
	int level = atoi(argv[3]);
        convert_dataset(string(argv[1]), string(argv[2]), level);
    }

    return 0;
}
