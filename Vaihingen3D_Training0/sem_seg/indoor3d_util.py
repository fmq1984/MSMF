#-*- coding: utf-8 -*-
import numpy as np
import glob
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

DATA_PATH = os.path.join(ROOT_DIR, 'data', 'Vaihingen3D_Training')
g_classes = [x.rstrip() for x in open(os.path.join(BASE_DIR, 'meta/class_names.txt'))]
g_class2label = {cls: i for i,cls in enumerate(g_classes)}#enumerate 用来遍历一个可迭代容器中的元素，同时通过一个计数器变量记录当前元素所对应的索引值。
g_class2color = {'Low_veg':	    [0,255,0],#绿   PointNet sem_seg代码笔记——里有
                 'Roof':	[0,0,255],#深蓝
                 'Car':         [255,0,0],#红
                 'Imp_surface':  [200,100,100],#红棕
                 'Tree':        [10,200,100],#浅绿
                 'Shrub':     [50,50,50],#黑
                 'Powerline':   [255,255,0],#黄
                 'Hedge':    [255,0,255],#紫
                 'Facade':    [200,200,100]}
g_easy_view_labels = [7,8,9,10,11,1]
g_label2color = {g_classes.index(cls): g_class2color[cls] for cls in g_classes}


# -----------------------------------------------------------------------------
# CONVERT ORIGINAL DATA TO OUR DATA_LABEL FILES
# -----------------------------------------------------------------------------

def collect_point_label(anno_path, out_filename, file_format='txt'):
    """ Convert original dataset files to data_label file (each line is XYZRGBL).
    数据转格式成 XYZRGBL
    Args:
        anno_path: path to annotations. e.g. Area_1/office_2/Annotations/
        out_filename: path to save collected points and labels (each line is XYZRGBL)
        file_format: txt or numpy, determines what file format to save.
    Returns:
        None
    Note:
        the points are shifted before save, the most negative point is now at origin.
    """
    points_list = []
    
    for f in glob.glob(os.path.join(anno_path, '*.txt')):
        """查找符合特定规则的文件路径名，返回所有匹配的文件路径列表"""
        cls = os.path.basename(f).split('.')[0]
        """将物品类名取出，例如：beam，board，board..."""
        """os.path.basename() 返回path最后的文件名"""
        if cls not in g_classes: # note: in some room there is 'staris' class..
            # g_classes = [x.strip() for x in open(os.path.join(...,"class_names.txt"))]
            cls = 'Facade'
        points = np.loadtxt(f)
        """np.loadtxt()读取txt文件，读入数据文件，要求每一行数据的格式相同，XYZRGB"""
        labels = np.ones((points.shape[0],1)) * g_class2label[cls]
        """比如one生成(点云数量,1) * 该类别的索引编号,即为该类别所有点云打上了标签"""
        # g_class2label = {cls: i for i,cls in enumerate(g_classes)} 将class_names中的类名分索引
        points_list.append(np.concatenate([points, labels], 1)) # Nx7，XYZRGBlabel 添加到列表末尾
    
    data_label = np.concatenate(points_list, 0)#数组拼接,xyz相对位置 + label
    xyz_min = np.amin(data_label, axis=0)[0:3]#取出所有坐标中的最小值
    """
        np.amin(a,axis)，返回数组中的最小值
    """
    data_label[:, 0:3] -= xyz_min
    """坐标全部减去最小坐标，全部移动至原点处"""
    
    if file_format=='txt':
        fout = open(out_filename, 'w')
        for i in range(data_label.shape[0]):
            fout.write('%f %f %f %d %d %d %d\n' % \
                          (data_label[i,0], data_label[i,1], data_label[i,2],
                           data_label[i,3], data_label[i,4], data_label[i,5],
                           data_label[i,6]))
        fout.close()
        """
            将房间中包含的所有类别的点云数据全部写入一个npy文本中，
        """
    elif file_format=='numpy':
        np.save(out_filename, data_label)
    else:
        print('ERROR!! Unknown file format: %s, please use txt or numpy.' % \
            (file_format))
        exit()

def point_label_to_obj(input_filename, out_filename, label_color=True, easy_view=False, no_wall=False):
    """ For visualization of a room from data_label file, XYZRGBL转obj
	input_filename: each line is X Y Z R G B L
	out_filename: OBJ filename,
            visualize input file by coloring point with label color
        easy_view: only visualize furnitures and floor
    """
    data_label = np.loadtxt(input_filename)
    data = data_label[:, 0:6]
    label = data_label[:, -1].astype(int)
    fout = open(out_filename, 'w')
    for i in range(data.shape[0]):
        color = g_label2color[label[i]]
        if easy_view and (label[i] not in g_easy_view_labels):
            continue
        if no_wall and ((label[i] == 2) or (label[i]==0)):
            continue
        if label_color:
            fout.write('v %f %f %f %d %d %d\n' % \
                (data[i,0], data[i,1], data[i,2], color[0], color[1], color[2]))
        else:
            fout.write('v %f %f %f %d %d %d\n' % \
                (data[i,0], data[i,1], data[i,2], data[i,3], data[i,4], data[i,5]))
    fout.close()
 


# -----------------------------------------------------------------------------
# PREPARE BLOCK DATA FOR DEEPNETS TRAINING/TESTING
# -----------------------------------------------------------------------------

def sample_data(data, num_sample):
    """ data is in N x ...
        we want to keep num_samplexC of them.
        if N > num_sample, we will randomly keep num_sample of them.
        if N < num_sample, we will randomly duplicate samples.
    """
    N = data.shape[0]
    if (N == num_sample):
        return data, range(N)
    elif (N > num_sample):
        sample = np.random.choice(N, num_sample)
        return data[sample, ...], sample
    else:
        sample = np.random.choice(N, num_sample-N)
        dup_data = data[sample, ...]
        return np.concatenate([data, dup_data], 0), list(range(N))+list(sample)

def sample_data_label(data, label, num_sample):
    new_data, sample_indices = sample_data(data, num_sample)
    new_label = label[sample_indices]
    return new_data, new_label
    
def room2blocks(data, label, num_point, block_size=5.0, stride=1.0,
                random_sample=False, sample_num=None, sample_aug=1):
    """ Prepare block training data.
    Args:
        data: N x 6 numpy array, 012 are XYZ in meters, 345 are RGB in [0,1]
            assumes the data is shifted (min point is origin) and aligned
            (aligned with XYZ axis)
        label: N size uint8 numpy array from 0-12
        num_point: int, how many points to sample in each block
        block_size: float, physical size of the block in meters
        stride: float, stride for block sweeping
        random_sample: bool, if True, we will randomly sample blocks in the room
        sample_num: int, if random sample, how many blocks to sample
            [default: room area]
        sample_aug: if random sample, how much aug
    Returns:
        block_datas: K x num_point x 6 np array of XYZRGB, RGB is in [0,1]
        block_labels: K x num_point x 1 np array of uint8 labels
        
    TODO: for this version, blocking is in fixed, non-overlapping pattern.
    """
    assert(stride<=block_size)

    limit = np.amax(data, 0)[0:3]#取出xyz的最大值，为确定一共有多少block
     
    # Get the corner location for our sampling blocks ，获取采样块的拐角位置
    xbeg_list = []
    ybeg_list = []
    if not random_sample:
        num_block_x = int(np.ceil((limit[0] - block_size) / stride)) + 1#因为collect data的时候indoor3d_util.collect_point_label()将最负点移到了原点，所以不用减去最小值
        ''' 
        np.ceil计算大于等于改值的最小整数
        x方向的分块数为6
        '''
        num_block_y = int(np.ceil((limit[1] - block_size) / stride)) + 1
        ''' 
        y方向的分块数为5
        '''
        for i in range(num_block_x):
            '''先定x坐标，按y方向逐个移动分块'''
            for j in range(num_block_y):
                xbeg_list.append(i*stride)
                ybeg_list.append(j*stride)
                '''将块索引填入列表中'''
    else:
        num_block_x = int(np.ceil(limit[0] / block_size))
        num_block_y = int(np.ceil(limit[1] / block_size))
        if sample_num is None:
            sample_num = num_block_x * num_block_y * sample_aug
        for _ in range(sample_num):
            xbeg = np.random.uniform(-block_size, limit[0]) 
            ybeg = np.random.uniform(-block_size, limit[1]) 
            xbeg_list.append(xbeg)
            ybeg_list.append(ybeg)

    # Collect blocks取出分好的块
    block_data_list = []
    block_label_list = []
    idx = 0
    for idx in range(len(xbeg_list)): 
       xbeg = xbeg_list[idx]
       ybeg = ybeg_list[idx]
       '''
       开始获取块中点云信息，做个判断，满足x坐标<1且同时x坐标>1为true，否则false
       满足y的坐标是否在此块中
       '''
       xcond = (data[:,0]<=xbeg+block_size) & (data[:,0]>=xbeg)
       ycond = (data[:,1]<=ybeg+block_size) & (data[:,1]>=ybeg)

       cond = xcond & ycond# xy坐标同时满足的为true，相当于确定了点的索引

       if np.sum(cond) < 100: # discard block if there are less than 100 pts.
           continue
       
       block_data = data[cond, :]# data包含所有点的xyzRGB信息，提取出符合该块范围的点云
       block_label = label[cond]#提取出对应的标签
       
       # randomly subsample data
       block_data_sampled, block_label_sampled = \
           sample_data_label(block_data, block_label, num_point)
       '''
       将每个块的点都采样为4096个点，
       块中多于4096的点进行随机选择4096个，块中少于4096进行重复点填充至4096
       '''
       block_data_list.append(np.expand_dims(block_data_sampled, 0))
       block_label_list.append(np.expand_dims(block_label_sampled, 0))
    '''
    block_data_list包含了所有块的点云信息，包含xyzrgb
      block_label_list包含了所有块的点云标签
    '''
    return np.concatenate(block_data_list, 0), \
           np.concatenate(block_label_list, 0)


def room2blocks_plus(data_label, num_point, block_size, stride,
                     random_sample, sample_num, sample_aug):
    """ room2block with input filename and RGB preprocessing.
    """
    data = data_label[:,0:6]
    data[:,3:6] /= 255.0
    label = data_label[:,-1].astype(np.uint8)
    
    return room2blocks(data, label, num_point, block_size, stride,
                       random_sample, sample_num, sample_aug)

#提取出npy点云信息
def room2blocks_wrapper(data_label_filename, num_point, block_size=1.0, stride=1.0,
                        random_sample=False, sample_num=None, sample_aug=1):
    if data_label_filename[-3:] == 'txt':
        data_label = np.loadtxt(data_label_filename)
    elif data_label_filename[-3:] == 'npy':
        data_label = np.load(data_label_filename)
        '''将房间里的所有点信息读取出，包含xyzrgbl，命为data_label
        '''
    else:
        print('Unknown file type! exiting.')
        exit()
    return room2blocks_plus(data_label, num_point, block_size, stride,
                            random_sample, sample_num, sample_aug)

"""将单个房间的所有信息读取出来后，进入下面的模块处理数据
"""
def room2blocks_plus_normalized(data_label, num_point, block_size, stride,
                                random_sample, sample_num, sample_aug):
    """ room2block, with input filename and RGB preprocessing.
        for each block centralize XYZ, add normalized XYZ as 678 channels
        具有输入文件名和RGB预处理。对于每个block集中XYZ，将归一化XYZ添加为678通道
    """
    data = data_label[:,0:6]#将所有点的xyzrgb读出，data为列表
    data[:,3:6] /= 255.0#将rgb进行归一化
    label = data_label[:,-1].astype(np.uint8)#提取出所有的label
    '''将最大的xyz取出，进行xyz归一化做准备
    '''
    max_room_x = max(data[:,0])
    max_room_y = max(data[:,1])
    max_room_z = max(data[:,2])

    '''data_batch包含所有块的点云信息，包含xyzrgb
       label_batch包含了所有块的点云标签
    '''
    data_batch, label_batch = room2blocks(data, label, num_point, block_size, stride,
                                          random_sample, sample_num, sample_aug)
    new_data_batch = np.zeros((data_batch.shape[0], num_point, 9)) #创建个（30,4096,9），将所有信息都填入新的数组中
    for b in range(data_batch.shape[0]):#一共有30个块，每个块进行读取
        new_data_batch[b, :, 6] = data_batch[b, :, 0]/max_room_x#将房间的归一化坐标填入678
        new_data_batch[b, :, 7] = data_batch[b, :, 1]/max_room_y#将xyz坐标除以最大xyz坐标作为归一化坐标
        new_data_batch[b, :, 8] = data_batch[b, :, 2]/max_room_z
        minx = min(data_batch[b, :, 0])#最小x坐标
        miny = min(data_batch[b, :, 1])#最小y坐标
        data_batch[b, :, 0] -= (minx+block_size/2) #把每一个block的xy平面的中心都移到原点
        data_batch[b, :, 1] -= (miny+block_size/2)
    new_data_batch[:, :, 0:6] = data_batch
    return new_data_batch, label_batch


'''提取出npy点云信息'''
def room2blocks_wrapper_normalized(data_label_filename, num_point, block_size=1.0, stride=1.0,
                                   random_sample=False, sample_num=None, sample_aug=1):
    if data_label_filename[-3:] == 'txt':
        data_label = np.loadtxt(data_label_filename)
    elif data_label_filename[-3:] == 'npy':
        data_label = np.load(data_label_filename)
        """ 将房间里的所有点信息读取出，包含xyzRGBL，命为data_label"""
    else:
        print('Unknown file type! exiting.')
        exit()
    return room2blocks_plus_normalized(data_label, num_point, block_size, stride,
                                       random_sample, sample_num, sample_aug)

def room2samples(data, label, sample_num_point):
    """ Prepare whole room samples.

    Args:
        data: N x 6 numpy array, 012 are XYZ in meters, 345 are RGB in [0,1]
            assumes the data is shifted (min point is origin) and
            aligned (aligned with XYZ axis)
        label: N size uint8 numpy array from 0-12
        sample_num_point: int, how many points to sample in each sample
    Returns:
        sample_datas: K x sample_num_point x 9
                     numpy array of XYZRGBX'Y'Z', RGB is in [0,1]
        sample_labels: K x sample_num_point x 1 np array of uint8 labels
    """
    N = data.shape[0]
    order = np.arange(N)
    np.random.shuffle(order) 
    data = data[order, :]
    label = label[order]

    batch_num = int(np.ceil(N / float(sample_num_point)))
    sample_datas = np.zeros((batch_num, sample_num_point, 6))
    sample_labels = np.zeros((batch_num, sample_num_point, 1))

    for i in range(batch_num):
        beg_idx = i*sample_num_point
        end_idx = min((i+1)*sample_num_point, N)
        num = end_idx - beg_idx
        sample_datas[i,0:num,:] = data[beg_idx:end_idx, :]
        sample_labels[i,0:num,0] = label[beg_idx:end_idx]
        if num < sample_num_point:
            makeup_indices = np.random.choice(N, sample_num_point - num)
            sample_datas[i,num:,:] = data[makeup_indices, :]
            sample_labels[i,num:,0] = label[makeup_indices]
    return sample_datas, sample_labels

def room2samples_plus_normalized(data_label, num_point):
    """ room2sample, with input filename and RGB preprocessing.
        for each block centralize XYZ, add normalized XYZ as 678 channels
    """
    data = data_label[:,0:6]
    data[:,3:6] /= 255.0 #颜色归一化到[0,1]
    label = data_label[:,-1].astype(np.uint8)
    max_room_x = max(data[:,0])
    max_room_y = max(data[:,1])
    max_room_z = max(data[:,2])
    #print(max_room_x, max_room_y, max_room_z)
    
    data_batch, label_batch = room2samples(data, label, num_point)
    new_data_batch = np.zeros((data_batch.shape[0], num_point, 9))
    for b in range(data_batch.shape[0]):
        new_data_batch[b, :, 6] = data_batch[b, :, 0]/max_room_x
        new_data_batch[b, :, 7] = data_batch[b, :, 1]/max_room_y
        new_data_batch[b, :, 8] = data_batch[b, :, 2]/max_room_z
        #minx = min(data_batch[b, :, 0])
        #miny = min(data_batch[b, :, 1])
        #data_batch[b, :, 0] -= (minx+block_size/2)
        #data_batch[b, :, 1] -= (miny+block_size/2)
    new_data_batch[:, :, 0:6] = data_batch
    return new_data_batch, label_batch


def room2samples_wrapper_normalized(data_label_filename, num_point):
    if data_label_filename[-3:] == 'txt':
        data_label = np.loadtxt(data_label_filename)
    elif data_label_filename[-3:] == 'npy':
        data_label = np.load(data_label_filename)
    else:
        print('Unknown file type! exiting.')
        exit()
    return room2samples_plus_normalized(data_label, num_point)


# -----------------------------------------------------------------------------
# EXTRACT INSTANCE BBOX FROM ORIGINAL DATA (for detection evaluation)
# -----------------------------------------------------------------------------

def collect_bounding_box(anno_path, out_filename):#提取边界
    """ Compute bounding boxes from each instance in original dataset files on
        one room. **We assume the bbox is aligned with XYZ coordinate.**
    
    Args:
        anno_path: path to annotations. e.g. Area_1/office_2/Annotations/
        out_filename: path to save instance bounding boxes for that room.
            each line is x1 y1 z1 x2 y2 z2 label,
            where (x1,y1,z1) is the point on the diagonal closer to origin
    Returns:
        None
    Note:
        room points are shifted, the most negative point is now at origin.
    """
    bbox_label_list = []

    for f in glob.glob(os.path.join(anno_path, '*.txt')):
        cls = os.path.basename(f).split('_')[0]
        if cls not in g_classes: # note: in some room there is 'staris' class..
            cls = 'clutter'
        points = np.loadtxt(f)
        label = g_class2label[cls]
        # Compute tightest axis aligned bounding box
        xyz_min = np.amin(points[:, 0:3], axis=0)
        xyz_max = np.amax(points[:, 0:3], axis=0)
        ins_bbox_label = np.expand_dims(
            np.concatenate([xyz_min, xyz_max, np.array([label])], 0), 0)
        bbox_label_list.append(ins_bbox_label)

    bbox_label = np.concatenate(bbox_label_list, 0)
    room_xyz_min = np.amin(bbox_label[:, 0:3], axis=0)
    bbox_label[:, 0:3] -= room_xyz_min 
    bbox_label[:, 3:6] -= room_xyz_min 

    fout = open(out_filename, 'w')
    for i in range(bbox_label.shape[0]):
        fout.write('%f %f %f %f %f %f %d\n' % \
                      (bbox_label[i,0], bbox_label[i,1], bbox_label[i,2],
                       bbox_label[i,3], bbox_label[i,4], bbox_label[i,5],
                       bbox_label[i,6]))
    fout.close()

def bbox_label_to_obj(input_filename, out_filename_prefix, easy_view=False):#边界可视化
    """ Visualization of bounding boxes.
    
    Args:
        input_filename: each line is x1 y1 z1 x2 y2 z2 label
        out_filename_prefix: OBJ filename prefix,
            visualize object by g_label2color
        easy_view: if True, only visualize furniture and floor
    Returns:
        output a list of OBJ file and MTL files with the same prefix
    """
    bbox_label = np.loadtxt(input_filename)
    bbox = bbox_label[:, 0:6]
    label = bbox_label[:, -1].astype(int)
    v_cnt = 0 # count vertex
    ins_cnt = 0 # count instance
    for i in range(bbox.shape[0]):
        if easy_view and (label[i] not in g_easy_view_labels):
            continue
        obj_filename = out_filename_prefix+'_'+g_classes[label[i]]+'_'+str(ins_cnt)+'.obj'
        mtl_filename = out_filename_prefix+'_'+g_classes[label[i]]+'_'+str(ins_cnt)+'.mtl'
        fout_obj = open(obj_filename, 'w')
        fout_mtl = open(mtl_filename, 'w')
        fout_obj.write('mtllib %s\n' % (os.path.basename(mtl_filename)))

        length = bbox[i, 3:6] - bbox[i, 0:3]
        a = length[0]
        b = length[1]
        c = length[2]
        x = bbox[i, 0]
        y = bbox[i, 1]
        z = bbox[i, 2]
        color = np.array(g_label2color[label[i]], dtype=float) / 255.0

        material = 'material%d' % (ins_cnt)
        fout_obj.write('usemtl %s\n' % (material))
        fout_obj.write('v %f %f %f\n' % (x,y,z+c))
        fout_obj.write('v %f %f %f\n' % (x,y+b,z+c))
        fout_obj.write('v %f %f %f\n' % (x+a,y+b,z+c))
        fout_obj.write('v %f %f %f\n' % (x+a,y,z+c))
        fout_obj.write('v %f %f %f\n' % (x,y,z))
        fout_obj.write('v %f %f %f\n' % (x,y+b,z))
        fout_obj.write('v %f %f %f\n' % (x+a,y+b,z))
        fout_obj.write('v %f %f %f\n' % (x+a,y,z))
        fout_obj.write('g default\n')
        v_cnt = 0 # for individual box
        fout_obj.write('f %d %d %d %d\n' % (4+v_cnt, 3+v_cnt, 2+v_cnt, 1+v_cnt))
        fout_obj.write('f %d %d %d %d\n' % (1+v_cnt, 2+v_cnt, 6+v_cnt, 5+v_cnt))
        fout_obj.write('f %d %d %d %d\n' % (7+v_cnt, 6+v_cnt, 2+v_cnt, 3+v_cnt))
        fout_obj.write('f %d %d %d %d\n' % (4+v_cnt, 8+v_cnt, 7+v_cnt, 3+v_cnt))
        fout_obj.write('f %d %d %d %d\n' % (5+v_cnt, 8+v_cnt, 4+v_cnt, 1+v_cnt))
        fout_obj.write('f %d %d %d %d\n' % (5+v_cnt, 6+v_cnt, 7+v_cnt, 8+v_cnt))
        fout_obj.write('\n')

        fout_mtl.write('newmtl %s\n' % (material))
        fout_mtl.write('Kd %f %f %f\n' % (color[0], color[1], color[2]))
        fout_mtl.write('\n')
        fout_obj.close()
        fout_mtl.close() 

        v_cnt += 8
        ins_cnt += 1

def bbox_label_to_obj_room(input_filename, out_filename_prefix, easy_view=False, permute=None, center=False, exclude_table=False):
    """ Visualization of bounding boxes.
    
    Args:
        input_filename: each line is x1 y1 z1 x2 y2 z2 label
        out_filename_prefix: OBJ filename prefix,
            visualize object by g_label2color
        easy_view: if True, only visualize furniture and floor
        permute: if not None, permute XYZ for rendering, e.g. [0 2 1]
        center: if True, move obj to have zero origin
    Returns:
        output a list of OBJ file and MTL files with the same prefix
    """
    bbox_label = np.loadtxt(input_filename)
    bbox = bbox_label[:, 0:6]
    if permute is not None:
        assert(len(permute)==3)
        permute = np.array(permute)
        bbox[:,0:3] = bbox[:,permute]
        bbox[:,3:6] = bbox[:,permute+3]
    if center:
        xyz_max = np.amax(bbox[:,3:6], 0)
        bbox[:,0:3] -= (xyz_max/2.0)
        bbox[:,3:6] -= (xyz_max/2.0)
        bbox /= np.max(xyz_max/2.0)
    label = bbox_label[:, -1].astype(int)
    obj_filename = out_filename_prefix+'.obj' 
    mtl_filename = out_filename_prefix+'.mtl'

    fout_obj = open(obj_filename, 'w')
    fout_mtl = open(mtl_filename, 'w')
    fout_obj.write('mtllib %s\n' % (os.path.basename(mtl_filename)))
    v_cnt = 0 # count vertex
    ins_cnt = 0 # count instance
    for i in range(bbox.shape[0]):
        if easy_view and (label[i] not in g_easy_view_labels):
            continue
        if exclude_table and label[i] == g_classes.index('table'):
            continue

        length = bbox[i, 3:6] - bbox[i, 0:3]
        a = length[0]
        b = length[1]
        c = length[2]
        x = bbox[i, 0]
        y = bbox[i, 1]
        z = bbox[i, 2]
        color = np.array(g_label2color[label[i]], dtype=float) / 255.0

        material = 'material%d' % (ins_cnt)
        fout_obj.write('usemtl %s\n' % (material))
        fout_obj.write('v %f %f %f\n' % (x,y,z+c))
        fout_obj.write('v %f %f %f\n' % (x,y+b,z+c))
        fout_obj.write('v %f %f %f\n' % (x+a,y+b,z+c))
        fout_obj.write('v %f %f %f\n' % (x+a,y,z+c))
        fout_obj.write('v %f %f %f\n' % (x,y,z))
        fout_obj.write('v %f %f %f\n' % (x,y+b,z))
        fout_obj.write('v %f %f %f\n' % (x+a,y+b,z))
        fout_obj.write('v %f %f %f\n' % (x+a,y,z))
        fout_obj.write('g default\n')
        fout_obj.write('f %d %d %d %d\n' % (4+v_cnt, 3+v_cnt, 2+v_cnt, 1+v_cnt))
        fout_obj.write('f %d %d %d %d\n' % (1+v_cnt, 2+v_cnt, 6+v_cnt, 5+v_cnt))
        fout_obj.write('f %d %d %d %d\n' % (7+v_cnt, 6+v_cnt, 2+v_cnt, 3+v_cnt))
        fout_obj.write('f %d %d %d %d\n' % (4+v_cnt, 8+v_cnt, 7+v_cnt, 3+v_cnt))
        fout_obj.write('f %d %d %d %d\n' % (5+v_cnt, 8+v_cnt, 4+v_cnt, 1+v_cnt))
        fout_obj.write('f %d %d %d %d\n' % (5+v_cnt, 6+v_cnt, 7+v_cnt, 8+v_cnt))
        fout_obj.write('\n')

        fout_mtl.write('newmtl %s\n' % (material))
        fout_mtl.write('Kd %f %f %f\n' % (color[0], color[1], color[2]))
        fout_mtl.write('\n')

        v_cnt += 8
        ins_cnt += 1

    fout_obj.close()
    fout_mtl.close() 


def collect_point_bounding_box(anno_path, out_filename, file_format):#坐标计算
    """ Compute bounding boxes from each instance in original dataset files on
        one room. **We assume the bbox is aligned with XYZ coordinate.**
        Save both the point XYZRGB and the bounding box for the point's
        parent element.
 
    Args:
        anno_path: path to annotations. e.g. Area_1/office_2/Annotations/
        out_filename: path to save instance bounding boxes for each point,
            plus the point's XYZRGBL
            each line is XYZRGBL offsetX offsetY offsetZ a b c,
            where cx = X+offsetX, cy=X+offsetY, cz=Z+offsetZ
            where (cx,cy,cz) is center of the box, a,b,c are distances from center
            to the surfaces of the box, i.e. x1 = cx-a, x2 = cx+a, y1=cy-b etc.
        file_format: output file format, txt or numpy
    Returns:
        None

    Note:
        room points are shifted, the most negative point is now at origin.
    """
    point_bbox_list = []

    for f in glob.glob(os.path.join(anno_path, '*.txt')):
        cls = os.path.basename(f).split('_')[0]
        if cls not in g_classes: # note: in some room there is 'staris' class..
            cls = 'clutter'
        points = np.loadtxt(f) # Nx6
        label = g_class2label[cls] # N,
        # Compute tightest axis aligned bounding box
        xyz_min = np.amin(points[:, 0:3], axis=0) # 3,
        xyz_max = np.amax(points[:, 0:3], axis=0) # 3,
        xyz_center = (xyz_min + xyz_max) / 2
        dimension = (xyz_max - xyz_min) / 2

        xyz_offsets = xyz_center - points[:,0:3] # Nx3
        dimensions = np.ones((points.shape[0],3)) * dimension # Nx3
        labels = np.ones((points.shape[0],1)) * label # N
        point_bbox_list.append(np.concatenate([points, labels,
                                           xyz_offsets, dimensions], 1)) # Nx13

    point_bbox = np.concatenate(point_bbox_list, 0) # KxNx13
    room_xyz_min = np.amin(point_bbox[:, 0:3], axis=0)
    point_bbox[:, 0:3] -= room_xyz_min 

    if file_format == 'txt':
        fout = open(out_filename, 'w')
        for i in range(point_bbox.shape[0]):
            fout.write('%f %f %f %d %d %d %d %f %f %f %f %f %f\n' % \
                          (point_bbox[i,0], point_bbox[i,1], point_bbox[i,2],
                           point_bbox[i,3], point_bbox[i,4], point_bbox[i,5],
                           point_bbox[i,6],
                           point_bbox[i,7], point_bbox[i,8], point_bbox[i,9],
                           point_bbox[i,10], point_bbox[i,11], point_bbox[i,12]))
        
        fout.close()
    elif file_format == 'numpy':
        np.save(out_filename, point_bbox)
    else:
        print('ERROR!! Unknown file format: %s, please use txt or numpy.' % \
            (file_format))
        exit()


