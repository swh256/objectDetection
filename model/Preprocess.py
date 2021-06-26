import pickle
import os
import mindspore.dataset as ds 
import numpy as np

batch_size = 200
#当前文件夹路径
path = '/mnt/c/Users/s/OneDrive/doc/course/ML/exp2/model'

#原始数据文件路径
DATA_DIR = "/mnt/c/Users/s/OneDrive/doc/course/ML/exp2/data/person/"
train_data =  ds.VOCDataset(DATA_DIR, task="Detection", usage="train", decode=True, shuffle=True).batch(batch_size,True)
test_data = ds.VOCDataset(DATA_DIR, task="Detection", usage="test", decode=True, shuffle=True).batch(batch_size,True)


