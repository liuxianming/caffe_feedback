#include <leveldb/db.h>
#include <google/protobuf/text_format.h>
#include <glog/logging.h>

//Opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <stdint.h>
#include <fstream>  // NOLINT(readability/streams)
#include <string>

#include <boost/filesystem.hpp>

#include "caffe/proto/caffe.pb.h"
#include "caffe/common.hpp"
#include "caffe/util/io.hpp"
using std::string;

void convert_leveldb(char* db_path, char* save_path){
  leveldb::DB* db;
  leveldb::Options options;
  options.create_if_missing = false;
  options.max_open_files = 100;

  LOG(INFO)<<"Opening leveldb " << string(db_path);

  leveldb::Status status = leveldb::DB::Open(
					     options, db_path, &db);

  //Check the status

  //Check the save path
  boost::filesystem::path save_dir(save_path);
  if(boost::filesystem::create_directories(save_dir)) {
    LOG(INFO)<<"Creating image save directory: "<<string(save_path);
  }

  leveldb::Iterator* it = db->NewIterator(leveldb::ReadOptions());
  for(it->SeekToFirst(); it->Valid(); it->Next()) {
    LOG(INFO)<<"Processing image #"<<it->key().ToString();
    caffe::Datum datum;
    datum.ParseFromString(it->value().ToString());
    //get filename
    char buffer[256];
    sprintf(buffer, "%s/%s.jpg", save_path, it->key().ToString().c_str());
    string filename(buffer);

    caffe::WriteDataToImage(filename, datum.channels(), datum.height(), datum.width(), datum.data().c_str());
  }
}

int main(int argc, char** argv) {
  if (argc !=3) {
    printf("This converts the leveldb into images\n"
	   "Usage\n"
	   "    convert_leveldb_images LEVELDB SAVEPATH");
  } else {
    google::InitGoogleLogging(argv[0]);
    convert_leveldb(argv[1], argv[2]);
  }
}
