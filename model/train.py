from mindspore.nn.metrics import Accuracy
from mindspore.train.callback import LossMonitor
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
import utils
import os
import mindspore.nn as nn
from Parameter import Parameter
from model import Mynet
from Loss import MyLoss
from mindspore.nn.metrics import Accuracy
from mindspore.train.callback import LossMonitor
from mindspore.train import Model
from mindspore import load_checkpoint, load_param_into_net
# import progressbar
import mindspore.dataset as ds
import pickle
import argparse
import mindspore
from mindspore import context
# p = progressbar.ProgressBar()
def train_net( model, epoch_size, ds_path, param, ckpoint_cb):
    """define the training method"""
    print("============== Starting Training ==============")
   
    # generate data
    # utils.VOC2pkl(ds_path,'train',param)

     #load training dataset

    dataset_generator = utils.DatasetGenerator('./model/pkl/train.pkl')
    data = ds.GeneratorDataset(
        dataset_generator, ["image", "label"], shuffle=True)

        #set batch_size
    a = 0
    for i in data.create_dict_iterator():
        print('get size')
        print(i['image'].shape)
        break
        
    # print(a)
    data = data.batch(param.batch_size)
    
    model.train(epoch_size, data, callbacks=[ckpoint_cb, LossMonitor()])
if __name__ == '__main__':
    print(os.getcwd())
    # parser = argparse.ArgumentParser(description='MindSpore LeNet Example')
    # parser.add_argument('--device_target', type=str, default="CPU", choices=['Ascend', 'GPU', 'CPU'])

    # args = parser.parse_known_args()[0]
    # context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)



    param = Parameter()
        # set parameters of check point
    config_ck = CheckpointConfig(save_checkpoint_steps=20, keep_checkpoint_max=1) 
    # apply parameters of check point
    ckpoint_cb = ModelCheckpoint(prefix="Mynet", directory='./model/ckPoint', config=config_ck)
    epoch_size = 1    
    ds_path = "./data/person"
     #learning rate setting
    lr = 0.01
    momentum = 0.9
    #create the network
    network = Mynet()
    # # 将模型参数存入parameter的字典中
    # param_dict = load_checkpoint("./model/ckPoint/Mynet-1_40.ckpt")
    # # 将参数加载到网络中
    # load_param_into_net(network, param_dict)
    net_loss =  MyLoss()
    #define the optimizer
    # print(network.trainable_params())
    net_opt = nn.Adam(network.trainable_params(), lr,weight_decay=0.0005)
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})
    train_net( model, epoch_size, ds_path, param, ckpoint_cb)