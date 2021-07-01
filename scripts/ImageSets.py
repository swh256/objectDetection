# -*- coding: utf-8 -*-
'''
划分train、test等数据集，在./data/person/ImageSets/Main中生成txt文件
'''
import os
import random

# ==================可能需要修改的地方=====================================#
path = "C:/Users/s/OneDrive/doc/course/ML/exp2"
os.chdir(path)
g_root_path = "./data/person/"
xmlfilepath = "Annotations"  # 标注文件存放路径
saveBasePath = "ImageSets/Main/"  # ImageSets信息生成路径
trainval_percent = 0.99
train_percent = 0.99
# ==================可能需要修改的地方=====================================#

os.chdir(g_root_path)

total_xml = os.listdir(xmlfilepath)
num = len(total_xml)
xml_list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(xml_list, tv)
train = random.sample(trainval, tr)

print("train and val size", tv)
print("train  size", tr)
ftrainval = open(saveBasePath + "trainval.txt", "w")
ftest = open(saveBasePath + "test.txt", "w")
ftrain = open(saveBasePath + "train.txt", "w")
fval = open(saveBasePath + "val.txt", "w")

for i in xml_list:
    name = total_xml[i][:-4] + "\n"
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
