import matplotlib
# Set the backend of matplotlib
matplotlib.use('Agg')
import scipy
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig

# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Begin of class definition
class caffe_visualizer(object):

    # data member
    # Using variable net to store the network
    __net = object()
    __modelpath = (caffe_root + 'examples/imagenet/imagenet_deploy.prototxt', \
                        caffe_root + 'examples/imagenet/caffe_reference_imagenet_model', \
                        caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy' \
                        )

    # Initialization of class
    # modelpath is a tuple: modelpath[0] - deploy, modelpath[1] - trained model, modelpath[2] - mean value
    def __init__(self, inputpath=''):
        plt.rcParams['figure.figsize'] = (10, 10)
        plt.rcParams['image.interpolation'] = 'nearest'
        plt.rcParams['image.cmap'] = 'gray'

        self.load_net(inputpath)

    
    # The function used to load trained network from file    
    def load_net(self, inputpath):
        if inputpath:
            print 'Initialize using input path...'
            self.__modelpath = inputpath
        
        print self.__modelpath
        # initialize the network with specific path
        self.__net = caffe.Classifier(self.__modelpath[0], self.__modelpath[1])
        self.__net.set_phase_test()
        self.__net.set_mode_cpu()
        self.__net.set_mean('data', self.__modelpath[2])
        self.__net.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
        self.__net.set_input_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]

        # show network structure
        self.print_net_information()


    def print_net_information(self):
        # print the basic information of network
        print '--------------------------------------------'
        print '[Load network model from]:\n\tDeploy: %s\n\tModel: %s\n\tMeanValue: %s'%self.__modelpath
        print '--------------------------------------------'
        print '[Network Structure]:'
        for k,v in self.__net.params.items():
            print (k, v[0].data.shape) 
        print '--------------------------------------------'

    # The network takes BGR images, so we need to switch color channels
    def showimage(self, im):
        pl.clf()
        if im.ndim == 3:
            im = im[:, :, ::-1]
        pl.imshow(im)

    def plotimage(self, im, savepath=''):
        if im.ndim == 3:
            im = im[:, :, ::-1]
        if not savepath:
            self.showimage(im)
        else:
            # Instead of showing image, the following code just plot the visualization into a file
            pl.imshow(im)
            pl.savefig(savepath)
        pl.clf()


    # take an array of shape (n, height, width) or (n, height, width, channels)
    # and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
    def vis_square(self, data, savepath ='',  padsize=1, padval=0):
        data -= data.min()
        data /= data.max()
        
        # force the number of filters to be square
        n = int(np.ceil(np.sqrt(data.shape[0])))
        padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
        data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
        
        # tile the filters into an image
        data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
        data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
        
        if savepath == '' :
            self.showimage(data)
        else:
            self.plotimage(data, savepath)

    def run_test_img(self, imgpath):
        scores = self.__net.predict([caffe.io.load_image(imgpath)])
        return scores
    
    # Show/ Print to file: outputs (blobs in caffe) of given 'layer', from startidx - endidx
    # parameter num indicates which input window will be plotted
    # By default, the input window is 0,
    # This is caused by the input layer of _net.predict, which cropped and mirrored the input image
    # and leads to 10 windows
    def visualize_outputs(self, layer, num = 0, startidx = 0, endidx = 0, savepath = '', imgpath = ''):
        if imgpath:
            scores = self.run_test_img(imgpath)
            # print scores
        if endidx == 0:
            feat = self.__net.blobs[layer].data[num, startidx:]
        else:
            feat = self.__net.blobs[layer].data[num, startidx:endidx]
        self.vis_square(feat, savepath)

    # Show/ Print to file: filters (params in caffe) of given 'layer', from startidx - endidx
    def visualize_filters(self, layer, startidx = 0, endidx = 0, savepath = ''):
        filters = self.__net.params[layer][0].data
        # find the shape / size of filters
        layer_shape = self.get_params_layer_size(layer)
        if endidx == 0:
            filters = filters[startidx:].reshape((layer_shape[0] - startidx)*layer_shape[1],layer_shape[2],layer_shape[3])
        else:
            filters = filters[startidx:endidx].reshape((endidx - startidx)*layer_shape[1],layer_shape[2],layer_shape[3])
        # print layer_shape
        self.vis_square(filters,savepath)


    # Get the size of a given layer in the network
    def get_params_layer_size(self, layer):
        return self.__net.params[layer][0].data.shape

    def get_blobs_layer_size(self, layer):
        return self.__net.blobs[layer].data[4].shape
        
if __name__ == "__main__":
    # Start test
    print 'start testing...'
    vis = caffe_visualizer()
    vis.run_test_img('test.jpg')
    #vis.visualize_filters('conv1',savepath = 'conv1_filter.png')
    vis.visualize_outputs('conv1',endidx=36, savepath = 'conv1_output.png')
    #vis.visualize_filters('conv1')
    #vis.visualize_outputs('conv1',endidx=36, imgpath='test.jpg')
    
    print "[Done]"