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
import argparse
# p = progressbar.ProgressBar()

def train(model,pklPath,epoch_size,batch_size,ckpoint_cb):
    dataset_generator = utils.DatasetGenerator(pklPath)
    data = ds.GeneratorDataset(
        dataset_generator, ["image", "label"], shuffle=False)
    data = data.batch(batch_size)
        #set batch_size
    model.train(epoch_size, data,dataset_sink_mode=False, callbacks=[ckpoint_cb, LossMonitor()])

# def train_net( model, epoch_size, ds_path, param, ckpoint_cb):
#     """define the training method"""
#     print("============== Starting Training ==============")
   


#      #load training dataset

#     dataset_generator = utils.DatasetGenerator('./model/pkl/test.pkl')
#     data = ds.GeneratorDataset(
#         dataset_generator, ["image", "label"], shuffle=False)
#     data = data.batch(4)
#         #set batch_size
#     a = 0
#     # for i in data.create_dict_iterator():
#     #     print(i['image'].shape)
#     #     print(i['label'].shape)
#     #     break
        
        
#     # print(a)

    
#     model.train(epoch_size, data,dataset_sink_mode=False, callbacks=[ckpoint_cb, LossMonitor()])
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--workbash', type=str, default = None)
    parser.add_argument('--ds_name',type=str, default=None)
    parser.add_argument('--ckpt',type=str, default=None)
    # parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    ds.config.set_num_parallel_workers(8)
    path=args.workbash
    os.chdir(path)
        # set parameters of check point
    config_ck = CheckpointConfig(save_checkpoint_steps=5, keep_checkpoint_max=1) 
    # apply parameters of check point
    ckpoint_cb = ModelCheckpoint(prefix="Mynet", directory='./model/ckPoint', config=config_ck)
    ds_path = "./data/person"
    pklPath = "./model/pkl/train/"
     #learning rate setting
    lr = 0.01
    momentum = 0.9
    #create the network
    network = Mynet()
    net_loss =  MyLoss()
    #define the optimizer
    # print(network.trainable_params())
    net_opt = nn.Adam(network.trainable_params(), lr,weight_decay=0.0005)
    # generate data
    # utils.VOC2pkl(ds_path,'test')
    param_dict = load_checkpoint("./model/ckPoint/" + args.ckpt)
    # for f in os.listdir(pklPath):
        # try:
    ckpt = args.ckpt
    # 将模型参数存入parameter的字典中
    param_dict = load_checkpoint("./model/ckPoint/" + ckpt)
    # 将参数加载到网络中
    load_param_into_net(network, param_dict)


    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})
    train(model,pklPath + args.ds_name, 1,4,ckpoint_cb)
        #删除多余ckpt文件
            
        # except:
        #     print('err\n')
