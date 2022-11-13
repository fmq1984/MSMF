#coding=utf-8
import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))#使用sys.path.append()方法可以添加自定义的路径。
import provider
import tf_util
from model import *
from datetime import datetime

#使用 argparse 的第一步是创建一个 ArgumentParser 解析对象。ArgumentParser 对象包含将命令行解析成 Python 数据类型所需的全部信息。
parser = argparse.ArgumentParser()
#添加参数，增加属性
parser.add_argument('--gpu', type=int, default=1, help='GPU to use [default: GPU 0]')#help参数命令的介绍；默认不使用GPU

parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=4096, help='Point number [default: 4096]')
parser.add_argument('--max_epoch', type=int, default=50, help='Epoch to run [default: 50]')
parser.add_argument('--batch_size', type=int, default=24, help='Batch Size during training [default: 24]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')#default表默认
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=300000, help='Decay step for lr decay [default: 300000]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
FLAGS = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
NUM_POINT = FLAGS.num_point
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step#计算learning rate的衰减
DECAY_RATE = FLAGS.decay_rate

#记录训练日志，以及备份 model.py 和 train.py。
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp model.py %s' % (LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

MAX_NUM_POINT = 4096
NUM_CLASSES = 9

# BN_ 开头的4个变量用来计算 Batch Normalization 的Decay参数，即decay参数也随着训练逐渐decay。
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
#BN_DECAY_DECAY_STEP = float(DECAY_STEP * 2)
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

'''
原作者所用数据集划分为了24个h5格式的文件，名字存在all_files.txt 中
获取所有数据文件名
获取每个样本（Block）所对应的room
'''
ALL_FILES = provider.getDataFiles(os.path.join(ROOT_DIR,'data/Vaihingen3D_Training_hdf5_data/all_files.txt'))
room_filelist = [line.rstrip() for line in open(os.path.join(ROOT_DIR,'data/Vaihingen3D_Training_hdf5_data/room_filelist.txt'))] # room_filelist.txt 一共 23585 行; 即对应每个物体在哪个room采集的

# Load ALL data
#首先将数据定义为list，然后在循环中依次append，循环结束后再将list转换为numpy数组
data_batch_list = []
label_batch_list = []
for h5_filename in ALL_FILES:
    data_batch, label_batch = provider.loadDataFile(os.path.join(ROOT_DIR,'data',h5_filename))
    data_batch_list.append(data_batch)
    label_batch_list.append(label_batch)
data_batches = np.concatenate(data_batch_list, 0)
label_batches = np.concatenate(label_batch_list, 0)
print(data_batches.shape)
print(label_batches.shape)

#分配训练集和测试集
test_area = 'Area_'+str(5)#test_area 为从命令行解析的参数，原文数据集从6个区域中采样而得，训练时需指定哪一个区域的数据用来测试
train_idxs = []
test_idxs = []
for i,room_name in enumerate(room_filelist):# enumerate(): 用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
    if test_area in room_name:
        test_idxs.append(i)
    else:
        train_idxs.append(i)

train_data = data_batches[train_idxs,...]
train_label = label_batches[train_idxs]
test_data = data_batches[test_idxs,...]
test_label = label_batches[test_idxs]
print(train_data.shape, train_label.shape)
print(test_data.shape, test_label.shape)




def log_string(out_str):#用来log训练日志。
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):#计算指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    # 训练时学习率最好随着训练衰减，learning_rate最大取0.00001 (衰减后的学习率和0.00001取最大)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!!
    return learning_rate        

def get_bn_decay(batch):#计算衰减的Batch Normalization 的 decay
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

'''训练函数train'''
def train():
    with tf.Graph().as_default():#表示将这个类实例，也就是新生成的图作为整个 tensorflow
        with tf.device('/gpu:'+str(GPU_INDEX)):#如果需要切换成GPU运算，可以调用
            pointclouds_pl, labels_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize. global step 参数 初始化 为0, 每次自动加 1
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            '''
            由get_model()可知，pred的维度为B×N×13，13为Channel数，对应13个分类标签。
            每个点的这13个值最大的一个的下标即为所预测的分类标签。
            '''
            pred = get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)#预测值为pred，调用model.py 中的 get_model()得到
            loss = get_loss(pred, labels_pl)
            tf.summary.scalar('loss', loss)
            # tf.argmax(pred, 2) 返回pred C 这个维度的最大值索引
            # tf.equal() 比较两个张量对应位置是否想等，返回相同维度的bool值矩阵

            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            #获得衰减后的学习率，以及选择优化器optimizer
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True#让TensorFlow在运行过程中动态申请显存，避免过多的显存占用
        config.allow_soft_placement = True#当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行。
        config.log_device_placement = True#在终端打印出各项操作是在哪个设备上运行的。
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        #初始化参数，开始训练
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl:True})

        # ops 是一个字典，作为接口传入训练和评估 epoch 循环中。
        #  pred 是数据处理网络模块；loss 是 损失函数；train_op 是优化器；batch 是当前的批次
        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()    #在同一个位置刷新输出
             
            train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops, test_writer)#用来每运行一个epoch后evaluate在测试集的accuracy和loss
            
            # Save the variables to disk.每10个epoch保存1次模型。
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)
# --------------------------------------
# ----- 训练函数 train() 结束 -----
# --------------------------------------


def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    #log_string(str(datetime.now()))
    
    log_string('----')
    current_data, current_label, _ = provider.shuffle_data(train_data[:,0:NUM_POINT,:], train_label) 
    
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE#计算在指定BATCH_SIZE下，训练1个epoch 需要几个mini-batch训练。// 除完后对结果进行自动floor向下取整操作
    
    total_correct = 0
    total_seen = 0
    loss_sum = 0


    #batch_data = provider.rotate_point_cloud(current_data)
    #batch_data = provider.rotate_perturbation_point_cloud(batch_data)




 #在一个epoch 中逐个mini-batch训练直至遍历完一遍训练集。计算总分类正确数total_correct和已遍历样本数total_senn，总损失loss_sum.
    for batch_idx in range(num_batches):
        if batch_idx % 100 == 0:
            print('Current batch/total batch num: %d/%d'%(batch_idx,num_batches))
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        
        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred']],
                                         feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += loss_val

    log_string('mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('accuracy: %f' % (total_correct / float(total_seen)))
# -------------------------------------------------------------
# ----- train_one_epoch(sess, ops, train_writer) 函数 结束 -----
# -------------------------------------------------------------

        
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    
    log_string('----')
    current_data = test_data[:,0:NUM_POINT,:]
    current_label = np.squeeze(test_label)
    
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['loss'], ops['pred']],
                                      feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += (loss_val*BATCH_SIZE)
        for i in range(start_idx, end_idx):
            for j in range(NUM_POINT):
                l = current_label[i, j]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i-start_idx, j] == l)
            
    log_string('eval mean loss: %f' % (loss_sum / float(total_seen/NUM_POINT)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/(np.array(total_seen_class,dtype=np.float)+ 1e-6))))
         


if __name__ == "__main__":#这段代码确保只有单独运行testone.py时才会执行text()函数
    train()
    LOG_FOUT.close()
