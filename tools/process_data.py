#! ~/anaconda/bin/python

import sys
import cv2
import os
import shutil
import argparse
import numpy as np

def CropImage(imgPath, rect, size):
    im = cv2.imread(imgPath)
    # crop subimage from the bounding box
    # Bounding box is in format of: x, y, width, height
    subimg = im[rect[1]:(rect[1]+rect[3]),rect[0]:(rect[0]+rect[2])]
    # resize image
    resizedImg = cv2.resize(subimg, size)

    return resizedImg

def main(argv):
    parser = argparse.ArgumentParser(description='Prepare CUB Bird dataset for Caffe')
    if len(argv) != 3:
        print 'Usage: %s source_dir des_dir' %argv[0]
        return
    else:
        sourcePath = argv[1]
        desPath = argv[2]

        if not os.path.exists(desPath):
            os.mkdir(desPath)

        try:
            imageListFn = open(os.path.join(sourcePath, 'images.txt'),'r')
            classListFn = open(os.path.join(sourcePath, 'image_class_labels.txt'),'r')
            BBoxListFn = open(os.path.join(sourcePath, 'bounding_boxes.txt'), 'r')
            TrnTestSplitFn = open(os.path.join(sourcePath, 'train_test_split.txt'), 'r')

            trainingListFn = open(os.path.join(desPath, 'training.txt'),'w')
            testingListFn = open(os.path.join(desPath, 'testing.txt'),'w')

            # Creat dirs for stroing training and testing images
            # Training images and cropped_training images
            # cropped_training images are those cropped by bouding boxes
            if not os.path.exists(os.path.join(desPath, 'training')):
                os.mkdir(os.path.join(desPath, 'training'))
            if not os.path.exists(os.path.join(desPath, 'testing')):
                os.mkdir(os.path.join(desPath, 'testing'))
            if not os.path.exists(os.path.join(desPath, 'cropped_training')):
                os.mkdir(os.path.join(desPath, 'cropped_training'))
            if not os.path.exists(os.path.join(desPath, 'cropped_testing')):
                os.mkdir(os.path.join(desPath, 'cropped_testing'))

            # define the resized image size
            size = (224, 224,)
            
            # begin reading files:
            while True:
                line = imageListFn.readline()
                
                if not line:
                    break
                # Parsing txt files
                line = line.strip("\n")
                splitedStr = line.split(" ")
                imgID = splitedStr[0]
                imgPath = splitedStr[1]
                imgName = imgPath.split("/")[1]

                classLine = classListFn.readline().strip("\n")
                classID = classLine.split(" ")[1]

                print "Processing image:" + imgName + "...\n"

                BBoxLine = BBoxListFn.readline().strip("\n")
                BBoxStr = BBoxLine.split(" ")[1:5]
                # rect = [ int(float(BBoxStr[0])), int(float(BBoxStr[1])), int(float(BBoxStr[2])), int(float(BBoxStr[3])) ]
                rect = [ int(float(t)) for t in BBoxStr]

                # Crop + resize image and save to "cropped_training" dir
                croppedImg = CropImage(os.path.join(sourcePath, 'images', imgPath), rect, size)

                # Read whether or not the image is training or testing
                TrnFlag = int(TrnTestSplitFn.readline().strip('\n').split(" ")[1])

                if TrnFlag:
                    # The image is training image
                    cv2.imwrite(os.path.join(desPath, 'cropped_training', imgName), croppedImg)
                    # Copy the original image to "training" dir
                    shutil.copy(os.path.join(sourcePath, 'images', imgPath), os.path.join(desPath, 'training', imgName))

                    # Write "image_path class_id" to file
                    trainingListFn.write(imgName + " " + classID + "\n")
                else:
                    # The image is testing images
                    cv2.imwrite(os.path.join(desPath, 'cropped_testing', imgName), croppedImg)
                    # Copy the original image to "testing" dir
                    shutil.copy(os.path.join(sourcePath, 'images', imgPath), os.path.join(desPath, 'testing', imgName))

                    # Write "image_path class_id" to file
                    testingListFn.write(imgName + " " + classID + "\n")

            # Remember to close all the files
            imageListFn.close()
            classListFn.close()
            BBoxListFn.close()
            TrnTestSplitFn.close()
            trainingListFn.close()
            testingListFn.close()
            
        except IOError:
            print 'File Operation error'
    
if __name__ == '__main__':
    main(sys.argv)
