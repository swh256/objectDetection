import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore import Tensor
import pickle
import utils
from Parameter import Parameter
import mindspore.dataset as ds
import argparse
from mindspore import context
import mindspore
import os

class Mynet(nn.Cell):
    def __init__(self):
        super(Mynet, self).__init__()
        self.lrelu = nn.LeakyReLU(0.1)
        self.max_pool2d = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64, kernel_size=7,stride=2) # max pool
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=192, kernel_size=3,stride=1) # max pool
        self.conv3 = nn.Conv2d(in_channels=192,out_channels=128, kernel_size=1,stride=1)
        self.conv4 = nn.Conv2d(in_channels=128,out_channels=256, kernel_size=3,stride=1)
        self.conv5 = nn.Conv2d(in_channels=256,out_channels=256, kernel_size=1,stride=1)
        self.conv6 = nn.Conv2d(in_channels=256,out_channels=512, kernel_size=3,stride=1)#max pool 

        self.conv7 = nn.Conv2d(in_channels=512,out_channels=256, kernel_size=1,stride=1)
        self.conv8 = nn.Conv2d(in_channels=256,out_channels=512, kernel_size=3,stride=1)
        # multiply 4
        self.conv9 = nn.Conv2d(in_channels=512,out_channels=512, kernel_size=1,stride=1)
        self.conv10 = nn.Conv2d(in_channels=512,out_channels=1024, kernel_size=3,stride=1)# max pool

        self.conv11 = nn.Conv2d(in_channels=1024,out_channels=512, kernel_size=1,stride=1)
        self.conv12 = nn.Conv2d(in_channels=512,out_channels=1024, kernel_size=3,stride=1)
        # multiply 2
        self.conv13 = nn.Conv2d(in_channels=1024,out_channels=1024, kernel_size=3,stride=1)
        self.conv14 = nn.Conv2d(in_channels=1024,out_channels=1024, kernel_size=3,stride=2)
        self.conv15 = nn.Conv2d(in_channels=1024,out_channels=1024, kernel_size=3,stride=1)
        self.conv16= nn.Conv2d(in_channels=1024,out_channels=1024, kernel_size=3,stride=1)
        self.dense1 = nn.Dense(in_channels=7*7*1024,out_channels=4096)
        self.dense2 = nn.Dense(in_channels=4096,out_channels=7*7*11)
        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()

        self.fconv1 = nn.Conv2d(in_channels=512,out_channels=256, kernel_size=1,stride=1)
        self.norm9  = nn.BatchNorm2d(num_features=512)
        self.norm12  = nn.BatchNorm2d(num_features=1024)
        self.lineNormal = nn.BatchNorm1d(num_features=7*7*11)
        self.relu = nn.ReLU()
    def construct(self, x): 
        x = self.conv1(x)
        x = self.lrelu(x) 
        x = self.max_pool2d(x) 
        x= self.conv2(x)
        x = self.lrelu(x)
        x = self.max_pool2d(x)

        x = self.conv3(x)
        x = self.lrelu(x)
        x = self.conv4(x)
        x = self.lrelu(x)
        # x = self.conv5(x)
        # x = self.lrelu(x)
        x = self.conv6(x)
        x = self.lrelu(x)
        x = self.max_pool2d(x)

        x = self.conv7(x)
        x = self.lrelu(x)
        x = self.conv8(x)
        x = self.lrelu(x)
        # x = self.conv7(x)
        # x = self.lrelu(x)
        # x = self.conv8(x)
        # x = self.lrelu(x)
        # x = self.conv7(x)
        # x = self.lrelu(x)
        # x = self.conv8(x)
        # x = self.lrelu(x)
        # x = self.conv7(x)
        # x = self.lrelu(x)
        # x = self.conv8(x)
        # x = self.lrelu(x)

        x = self.conv9(x)
        x = self.norm9(x)
        x = self.lrelu(x)
        # x = self.conv10(x)
        # x = self.lrelu(x)
        x = self.max_pool2d(x)

        # x = self.conv11(x)
        # x = self.lrelu(x)
        x = self.conv12(x)
        x = self.norm12(x)
        x = self.lrelu(x)
        x = self.max_pool2d(x)
        # x = self.conv11(x)
        # x = self.lrelu(x)
        # x = self.conv12(x)
        # x = self.lrelu(x)

        # x = self.conv13(x)
        # x = self.lrelu(x)
        # x = self.conv14(x)
        # x = self.lrelu(x)
        # x= self.conv15(x)
        # x = self.lrelu(x)
        # x= self.conv16(x)
        # x = self.lrelu(x)
  
        x = self.flatten(x)

        x = self.dense1(x)

        x = self.lrelu(x)
        x = self.dense2(x)
        # x = self.relu(x)
        # x = self.lineNormal(x)
        x = self.sigmoid(x)
        return x.reshape(-1, (5*2+1), 7, 7)

        #fast version
        # return x
        

if __name__ == '__main__':



    # parser = argparse.ArgumentParser(description='MindSpore LeNet Example')
    # parser.add_argument('--device_target', type=str, default="GPU", choices=['Ascend', 'GPU', 'CPU'])

    # args = parser.parse_known_args()[0]
    # context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    
    
    
    
    
    # para = Parameter(2)
    # pklPath = '/home/swh/exp2/model/pkl/'
    # utils.VOC2pkl('/home/swh/exp2/data/person_test/',
    #     'test', Parameter(2))
    # pkl = []
    # with open(pklPath+'test.pkl', 'rb') as f:
    #     pkl = pickle.load(f)
    # # print(convert_bbox2labels(pkl[0]['label']))
    dataset_generator = utils.DatasetGenerator('./model/pkl/test.pkl')
    data = ds.GeneratorDataset(
        dataset_generator, ["image", "label"], shuffle=False)

    # net = Mynet()
    # conv1 = nn.Conv2d(in_channels=3,out_channels=64, kernel_size=7,stride=2)
    # conv2 = nn.Conv2d(in_channels=64,out_channels=192, kernel_size=3,stride=1)
    # relu = nn.ReLU()
    # max_pool2d = nn.MaxPool2d(kernel_size=2,stride=2)
    data = data.batch(2)
    # for da in data.create_dict_iterator():
    #     print(da['image'].shape)
    #     x = conv1(da['image'])
    #     x = relu(x)
    #     x = max_pool2d(x)
    #     x = conv2(x)
    #     x = relu(x)

    #     print(x.shape)

    mynet = Mynet()
    for da in data.create_dict_iterator():
        print((mynet.construct(da['image'])).shape)

    



