'''
此程序实现修改xml的folder标签和path标签
'''

import xml.etree.ElementTree as ET
import os

#xml文件路径
path = "/mnt/c/Users/s/OneDrive/doc/course/ML/exp2/data/person/Annotations"
# path = "/mnt/c/Users/s/OneDrive/doc/course/ML/exp2/testXml"
os.chdir(path)
files = os.listdir(path)
for file in files:
    tree = ET.parse(file)
    root = tree.getroot()
    folder = root.find('folder')
    path = root.find('path')
    folder.text = 'JPEGImages'
    path.text = '/mnt/c/Users/s/OneDrive/doc/course/ML/exp2/data/person/JPEGImages/'+ file

    tree.write(file)
