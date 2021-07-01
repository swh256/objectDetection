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
import mindspore.ops as ops

if __name__ == '__main__':
    print(os.getcwd())
    # parser = argparse.ArgumentParser(description='MindSpore LeNet Example')
    # parser.add_argument('--device_target', type=str, default="CPU", choices=['Ascend', 'GPU', 'CPU'])

    # args = parser.parse_known_args()[0]
    # context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    # param = Parameter()
    # utils.VOC2pkl('./data/person','test')
    dataset_generator = utils.DatasetGenerator('./model/pkl/test.pkl')
    data = ds.GeneratorDataset(
        dataset_generator, ["image", "label"], shuffle=True)

    data = data.batch(1)

    image = []
    label = []
    print('start generate image')
    for da in data.create_dict_iterator():
        # print(da['image'].shape)
        
        image.append(da['image'])
        label.append(da['label'])
        
    print('finish generate image')

    network = Mynet()
    # 将模型参数存入parameter的字典中
    param_dict = load_checkpoint("./model/ckPoint/Mynet-1_200.ckpt")
    # 将参数加载到网络中
    load_param_into_net(network, param_dict)

    #define the optimizer
    # print(network.trainable_params())

    model = Model(network)
    res = []
    for img in image:
        res.append(model.predict(img))
    print('test finish')

    for r in res:
      
        for img in image:
            utils.show_img(img,r)
            break
            
        
