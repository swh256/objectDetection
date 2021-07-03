import mindspore.dataset as ds
import numpy as np
from Parameter import Parameter
import pickle
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.vision.py_transforms as pv
import matplotlib.pyplot as plt
from mindspore.dataset.transforms.py_transforms import Compose
import mindspore
import torch
import mindspore.ops as ops
import cv2
import time
CLASS = ['person']
pklPath = './model/pkl/'

np.random.seed(58)


class DatasetGenerator:
    ''' 自定义构造数据集类
    '''

    def __init__(self, file):
        '''
        Args:
            is_train
        '''
        self.dataset = pickle.load(open(file, 'rb'))
        # print(self.dataset[0]['image'])

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, index):
        img = self.dataset[index]['image']

        bbox = self.dataset[index]['label']
  
        label = convert_bbox2labels(bbox)  

        return (img,label)

class ItrDatasetGenerator:
    ''' 自定义构造数据集类
    '''

    def __init__(self, file):
        '''
        Args:
            is_train
        '''
        self.dataset = pickle.load(open(file, 'rb'))
        # print(self.dataset[0]['image'])

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, index):
        img = self.dataset[index]['image']
        # cv2.imshow('', ops.Transpose()(img,(1,2,0)).asnumpy())
        # img = ops.Transpose()(img,(1,2,0)).asnumpy()
        # print(img.shape)
        # cv2.imshow('',img)
        # cv2.waitKey (0)
        label = self.dataset[index]['label']

        # label = ops.Transpose()(label,(2,0,1)).asnumpy()
        # print(label.shape)
        return (img,label)

def VOC2pkl(dir, _type):
    ''' 获取目的数据集
        获取train、test、val等pkl形式的数据集
        Args:
            dir (str):标准VOC数据集根路径
            type (str):取值为 train val test，指定要生成的数据类别
            para (Parameter): 参数对象
        Return:
            pkl :其中image是CHW形式的tensor
    '''
    print('************start generate data******************')
    _dataset = []  # 暂存数据
    info = {'width': 448, 'height': 448, 'name': 'person'}
    data = ds.VOCDataset(dir, task="Detection", usage=_type,
                         decode=True, shuffle=True)

    num = 0
    for item in data.create_dict_iterator():
        image,bbox = func(item['image'],item['bbox'])
        _data = {}
        _data['image'] = image

        #去除残缺数据
        if(len(bbox) < 4):
            continue
        _data['label'] = bbox2label(bbox, info)

        _dataset.append(_data)
    length = len(_dataset)
    #如果是测试集或者验证集，则分片保存
    if(_type == 'train' or _type=='val'):
        for i in range(int(np.ceil(length/20))):
            print("generate ",i,"th dataset")
            _subDs = _dataset[20*i:20*i+20]
            with open(pklPath+_type+ '/'+_type+str(i)+'.pkl', 'wb') as f:
                pickle.dump(_subDs, f)
    else:
        with open(pklPath+_type+ '/'+_type+'.pkl', 'wb') as f:
            pickle.dump(_dataset, f)
    # data = data.batch(para.batch_size, True)
    print('************generate data finished******************')

    # def func1(x):
    #     print('qq')
    #     return x
    # data = data.map(func1, input_columns=['image'])

    # transfList = Compose(
    #     [  func1, ])
    # # transfList = Compose(
    # #     [pv.Decode(), pv.ToPIL(), pv.Pad(padding=[padw, padh, padw, padh]), pv.Resize([input_size, input_size]), pv.ToTensor()])
    # data = data.map(transfList, input_columns=['image','bbox'])

    # # TODO 数据增广暂未做

    # itr = data.create_dict_iterator()
    # # TODO 外部传入info
    # print(type(itr))
    # for item in itr:
    #     print('enter itr')


def func(x,y):
    #image 处理，包括padding resize到448*448 RGB归一化 ->CHW tensor
    # _x = pv.Decode()(x)

    # _y = y.squeeze().asnumpy()
    # p1 = ((int)(_y[0]), (int)(_y[1]))
    # p2 = ((int)(_y[0]+_y[2]), (int)(_y[1]+_y[3]))
    # img = x.asnumpy()
    # cls_name = 'person'

    # cv2.rectangle(img,p1,p2,color=(255,0,0))
    # cv2.line(img,(0,0),(1280,720),color=(255,0,0))
    # cv2.circle(img,p1,102,color=(0,0,255))
    # cv2.circle(img,p2,10,color=(0,255,0))
    # cv2.putText(img,cls_name,p1,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))
    # cv2.imshow("bbox",img)
    # cv2.waitKey(0)

    _x = pv.ToPIL()(x.asnumpy())
    h,w = np.array(_x).shape[0:2]
    padw, padh = 0, 0
    if h > w:
        padw = (h - w) // 2
    elif w > h:
        padh = (w - h) // 2
    _x = pv.Pad(padding=[padw, padh, padw, padh])(_x)
    _x = pv.Resize([448, 448])(_x)
    _x = pv.ToTensor()(_x)
    _x = np.array(_x)

    #label 处理，从(xmin,ymin.xmax,ymax) => (xmin,ymin,h,w),但未处理len(bbox) < 4的情况
    _y = y.squeeze().asnumpy()

    if(len(_y) == 4):
        m = max(h,w)
        _y = mindspore.Tensor(np.array([_y[0] * 448/m, (_y[1] + abs(w-h)/2)*448/m, _y[2]*448/m, _y[3]*448/m]), mindspore.float32)
        # _y = _y.asnumpy()
        # p1 = ((int)(_y[0]), (int)(_y[1]))
        # p2 = ((int)(_y[0]+_y[2]), (int)(_y[1]+_y[3]))
        # img = ops.Transpose()(_x,(1,2,0)).asnumpy()
        # cls_name = 'person'

        # cv2.rectangle(img,p1,p2,color=(255,0,0))
        # cv2.putText(img,cls_name,p1,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))
        # cv2.imshow("bbox",img)
        # cv2.waitKey(0)

    return (_x,_y)


def bbox2label(bbox, info):
    '''bbox -> (cls,xc,yc,w,h)并进行了归一化

        return:

    '''
    sbbox = np.squeeze(bbox)  # 去掉多余的维度
    dw = 1. / info['width']
    dh = 1. / info['height']
    x = ((2 * sbbox[0] + sbbox[2]) / 2 * dw).asnumpy()
    y = ((2 * sbbox[1] + sbbox[3]) / 2 * dh).asnumpy()
    w = (sbbox[2] * dw).asnumpy()
    h = (sbbox[3] * dh).asnumpy()
    cls = CLASS.index(info['name'])
    
    return [cls, x, y, w, h]


def convert_bbox2labels(bbox):
    '''
    convert (cls,x,y,w,h) to (7,7,5*B+cls)
    为了方便计算Loss
    注意，输入的bbox的信息是(xc,yc,w,h)格式的，转换为labels后，bbox的信息转换为了(px,py,w,h)格式
    '''

    gridsize = 1.0/7
    labels = np.zeros((7, 7, 5*2+1))  # 注意，此处需要根据不同数据集的类别个数进行修改

    gridx = int(bbox[1] // gridsize)  # 当前bbox中心落在第gridx个网格,列
    gridy = int(bbox[2] // gridsize)  # 当前bbox中心落在第gridy个网格,行
    # (bbox中心坐标 - 网格左上角点的坐标)/网格大小  ==> bbox中心点的相对位置
    gridpx = bbox[1] / gridsize - gridx
    gridpy = bbox[2] / gridsize - gridy
    # 将第gridy行，gridx列的网格设置为负责当前ground truth的预测，置信度和对应类别概率均置为1
    labels[gridy, gridx, 0:5] = np.array([gridpx, gridpy, bbox[3], bbox[4], 1])
    labels[gridy, gridx, 5:10] = np.array(
        [gridpx, gridpy, bbox[3], bbox[4], 1])
    labels[gridy, gridx, 10+int(bbox[0])] = 1
   
    return ops.Transpose()( mindspore.Tensor( labels,mindspore.float32),(2,0,1)).asnumpy()




def labels2bbox(matrix):
    """
    将网络输出的7*7*30的数据转换为bbox的(98,25)的格式，然后再将NMS处理后的结果返回
    :param matrix: 注意，输入的数据中，bbox坐标的格式是(px,py,w,h)，需要转换为(x1,y1,x2,y2)的格式再输入NMS
    :return: 返回NMS处理后的结果
    """
    if matrix.size()[0:2]!=(7,7):
        raise ValueError("Error: Wrong labels size:",matrix.size())
    bbox = torch.zeros((98,25))
    # 先把7*7*30的数据转变为bbox的(98,25)的格式，其中，bbox信息格式从(px,py,w,h)转换为(x1,y1,x2,y2),方便计算iou
    for i in range(7):  # i是网格的行方向(y方向)
        for j in range(7):  # j是网格的列方向(x方向)
            bbox[2*(i*7+j),0:4] = torch.Tensor([(matrix[i, j, 0] + j) / 7 - matrix[i, j, 2] / 2,
                                                (matrix[i, j, 1] + i) / 7 - matrix[i, j, 3] / 2,
                                                (matrix[i, j, 0] + j) / 7 + matrix[i, j, 2] / 2,
                                                (matrix[i, j, 1] + i) / 7 + matrix[i, j, 3] / 2])
            bbox[2 * (i * 7 + j), 4] = matrix[i,j,4]
            bbox[2*(i*7+j),5:] = matrix[i,j,10:]
            bbox[2*(i*7+j)+1,0:4] = torch.Tensor([(matrix[i, j, 5] + j) / 7 - matrix[i, j, 7] / 2,
                                                (matrix[i, j, 6] + i) / 7 - matrix[i, j, 8] / 2,
                                                (matrix[i, j, 5] + j) / 7 + matrix[i, j, 7] / 2,
                                                (matrix[i, j, 6] + i) / 7 + matrix[i, j, 8] / 2])
            bbox[2 * (i * 7 + j)+1, 4] = matrix[i, j, 9]
            bbox[2*(i*7+j)+1,5:] = matrix[i,j,10:]
    print(bbox)
    return NMS(bbox)  # 对所有98个bbox执行NMS算法，清理cls-specific confidence score较低以及iou重合度过高的bbox


def NMS(bbox, conf_thresh=0.1, iou_thresh=0.3):
    """bbox数据格式是(n,25),前4个是(x1,y1,x2,y2)的坐标信息，第5个是置信度，后20个是类别概率
    :param conf_thresh: cls-specific confidence score的阈值
    :param iou_thresh: NMS算法中iou的阈值
    """
    n = bbox.size()[0]
    bbox_prob = bbox[:,5:].clone()  # 类别预测的条件概率
    bbox_confi = bbox[:, 4].clone().unsqueeze(1).expand_as(bbox_prob)  # 预测置信度
    bbox_cls_spec_conf = bbox_confi*bbox_prob  # 置信度*类别条件概率=cls-specific confidence score整合了是否有物体及是什么物体的两种信息
    bbox_cls_spec_conf[bbox_cls_spec_conf<=conf_thresh] = 0  # 将低于阈值的bbox忽略
    for c in range(20):
        rank = torch.sort(bbox_cls_spec_conf[:,c],descending=True).indices
        for i in range(98):
            if bbox_cls_spec_conf[rank[i],c]!=0:
                for j in range(i+1,98):
                    if bbox_cls_spec_conf[rank[j],c]!=0:
                        iou = calculate_iou(bbox[rank[i],0:4],bbox[rank[j],0:4])
                        if iou > iou_thresh:  # 根据iou进行非极大值抑制抑制
                            bbox_cls_spec_conf[rank[j],c] = 0
    bbox = bbox[torch.max(bbox_cls_spec_conf,dim=1).values>0]  # 将类别中最大的cls-specific confidence score为0的bbox都排除
    bbox_cls_spec_conf = bbox_cls_spec_conf[torch.max(bbox_cls_spec_conf,dim=1).values>0]
    res = torch.ones((bbox.size()[0],6))
    res[:,1:5] = bbox[:,0:4]  # 储存最后的bbox坐标信息
    res[:,0] = torch.argmax(bbox[:,5:],dim=1).int()  # 储存bbox对应的类别信息
    res[:,5] = torch.max(bbox_cls_spec_conf,dim=1).values  # 储存bbox对应的class-specific confidence scores
    return res


def calculate_iou(bbox1, bbox2):
    """计算bbox1=(x1,y1,x2,y2)和bbox2=(x3,y3,x4,y4)两个bbox的iou"""
    intersect_bbox = [0., 0., 0., 0.]  # bbox1和bbox2的交集
    if bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3]:
        pass
    else:
        intersect_bbox[0] = max(bbox1[0], bbox2[0])
        intersect_bbox[1] = max(bbox1[1], bbox2[1])
        intersect_bbox[2] = min(bbox1[2], bbox2[2])
        intersect_bbox[3] = min(bbox1[3], bbox2[3])

    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])  # bbox1面积
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])  # bbox2面积
    area_intersect = (intersect_bbox[2] - intersect_bbox[0]) * \
        (intersect_bbox[3] - intersect_bbox[1])  # 交集面积
    # print(bbox1,bbox2)
    # print(intersect_bbox)
    # input()

    if area_intersect > 0:
        return area_intersect / (area1 + area2 - area_intersect)  # 计算iou
    else:
        return 0


def show_img(img,label):
    ''' show img
        Args:
            
    '''
    print('start show img')

    bbox = labels2bbox(torch.Tensor(ops.Transpose()( label.squeeze(),(1,2,0)).asnumpy()))

    img = ops.Transpose()( img.squeeze(),(1,2,0)).asnumpy()
    h,w = img.shape[0:2]
    n = bbox.size()[0]
    print(n)
    for i in range(1):
        p1 = ((int)(w*bbox[i,1]), (int)(h*bbox[i,2]))
        p2 = ((int)(w*bbox[i,3]), (int)(h*bbox[i,4]))
        print(p1)
        cls_name = 'person'
        confidence = bbox[i,5]
        cv2.rectangle(img,p1,p2,color=(255,0,0))
        cv2.putText(img,cls_name,p1,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))
        cv2.imshow("bbox",img)
        cv2.waitKey(0)


if __name__ == '__main__':

    # VOC2pkl('./data/person/',
    #         'test', Parameter(2))
    # pkl = []
    # with open(pklPath+'test.pkl', 'rb') as f:
    #     pkl = pickle.load(f)
    # # print(convert_bbox2labels(pkl[0]['label']))
    param = Parameter()
    VOC2pkl('./data/person_test','test',param)
    dataset_generator = DatasetGenerator('./model/pkl/test.pkl')
    dataset = ds.GeneratorDataset(
        dataset_generator, ["image", "label"], shuffle=False)
    for data in dataset.create_dict_iterator():
        print(data['image'].shape)



        
    # a = np.ones((98,1))
    # b = np.ones((98,25))
    # b1 = b[:,4]
    # print(b1.shape)
    # print(np.expand_dims(b1,1).shape)
    # print( np.expand_dims( np.expand_dims(b1,1),(98,20)).shape)

        

   
    # img, label = pkl[0]['image'], pkl[0]['label']
    # h, w = img.shape[:2]
    # print(img)

    # pt1 = (int(label[1] * w - label[3] * w / 2),
    #        int(label[2] * h - label[4] * h / 2))
    # pt2 = (int(label[1] * w + label[3] * w / 2),
    #        int(label[2] * h + label[4] * h / 2))
    # cv2.putText(img, 'person', pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    # cv2.rectangle(img, pt1, pt2, (0, 0, 255, 2))
    # cv2.imwrite('my.jpg', img)
    # plt.imshow(img)
    # plt.savefig('my.jpg')

    # img = cv2.imread('/mnt/c/Users/s/OneDrive/doc/course/ML/exp2/data/person_test/' + "JPEGImages/" + '3200000100_20123132000600985967_0323_0_' + ".jpg")
    # h, w = img.shape[:2]
    # print(w,h)
    # label = [1,604,419,102,291]

    # pt1 = (int(label[1] ), int(label[2] ))
    # pt2 = (int(label[1] + label[3] ), int(label[2] + label[4]))
    # cv2.putText(img,'person',pt1,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))
    # cv2.rectangle(img,pt1,pt2,(0,0,255,2))

    # plt.imshow(img)
    # plt.savefig('img.jpg')