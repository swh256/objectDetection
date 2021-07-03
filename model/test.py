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
    #定义参数
    parser = argparse.ArgumentParser(description='MindSpore LeNet Example')
    parser.add_argument('--workbash', type=str, default = None)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--test_ds', type=str, default=None)
    args = parser.parse_args()

    path = args.workbash
    pklPath = "./model/pkl/test/"
    #切换工作目录
    os.chdir(path)
    #生成加载测试数据集
    dataset_generator = utils.DatasetGenerator(pklPath+args.test_ds)
    data = ds.GeneratorDataset(
        dataset_generator, ["image", "label"], shuffle=True)
    data = data.batch(1)

    #将image和label分隔开
    image = []
    label = []
    print('start generate image')
    for da in data.create_dict_iterator():
        image.append(da['image'])
        label.append(da['label'])  
    print('finish generate image')

    #定义网络
    network = Mynet()
    # 将模型参数存入parameter的字典中
    param_dict = load_checkpoint("./model/ckPoint/" + args.ckpt)
    # 将参数加载到网络中
    load_param_into_net(network, param_dict)

    model = Model(network)
    res = []
    for img in image[0:10]:
        res.append(model.predict(img))
    print('predict finish!')

    for index in range(len(res)):
    
        utils.show_img(image[index],res[index])


