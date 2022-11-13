#-*- coding: utf-8 -*-
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
import indoor3d_util

anno_paths = [line.rstrip() for line in open(os.path.join(BASE_DIR, 'meta/anno_paths.txt'))]#读取出每个房间点云文件的相对路径，Annotatations中包含了桌子椅子等类别的点云文件，包括xyz和RGB
anno_paths = [os.path.join(indoor3d_util.DATA_PATH, p) for p in anno_paths]#读取绝对路径

output_folder = os.path.join(ROOT_DIR, 'data/Vaihingen3D_Training_npy')
"""我创建了一个npy文件夹用于存放npy文件"""
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

for anno_path in anno_paths:
    '''将每一个Annotation文件夹中的文件读取出'''
    print(anno_path)
    #try:
    elements = anno_path.split('/')
    out_filename = elements[-3]+'_'+elements[-2]+'.npy' # Splice filename--Area_1_hallway_1.npy
    """获取npy文件名称"""
    indoor3d_util.collect_point_label(anno_path, os.path.join(output_folder, out_filename), 'numpy')
    """对应npy文件写入数据"""
    print(elements[-3],elements[-2])
    #except:
    #    print(anno_path, 'ERROR!!')
