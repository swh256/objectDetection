import cv2
import utils
from Parameter import Parameter
import mindspore
import mindspore.dataset as ds
import mindspore.ops as ops
import numpy as np
import pickle
import Loss
_y = [0,1,2,3]
#create dataset 
# print(len(mindspore.Tensor(np.array([_y[0], _y[1] , _y[2], _y[3]]), mindspore.float32)))
img =  cv2.imread('./data/person/JPEGImages/3202000001_21012232000601039837_0323_0_.jpg')

# cv2.imshow("Image", img)
# cv2.waitKey(0)

# pkl = pickle.load(open('./model/pkl/test.pkl','rb'))

# for i in pkl:
#     print(i['image'].shape)

# utils.VOC2pkl('./data/person','test')
print("VOC2PKl is ok!")
dataset_generator = utils.DatasetGenerator('./model/pkl/train.pkl')
dataset = ds.GeneratorDataset(dataset_generator, ["image", "label"], shuffle=True)
print("load dataset is ok!")
label  = []
dataset = dataset.batch(2)
for data in dataset.create_dict_iterator():
    print(data['image'].shape)
    print(data['label'].shape)
    label = data['label']
    break

loss = Loss.MyLoss()
print(label.asnumpy().shape)
print(loss(label,label))
    # cv2.imshow('',  ops.Transpose()(data['image'],(1,2,0)).asnumpy())
    # cv2.waitKey (0)
   


# print("generate img is ok!")

    

