# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import PIL.Image as Image
import random
import numpy as np
import cv2
import time

def get_frames_data(filename,seq_num,num_frames_per_clip=16):
    ''' Given a directory containing extracted frames, return a video clip of
    (num_frames_per_clip) consecutive frames as a list of np arrays '''
    ret_arr = []
    s_index = 0
    for parent, dirnames, filenames in os.walk(filename):
        #print(parent,dirnames,filenames)
        if(len(filenames)<num_frames_per_clip):
            return [], s_index
    filenames = sorted(filenames)
    #s_index = random.randint(0, len(filenames) - num_frames_per_clip)
    
    for s_index in range(0*num_frames_per_clip,seq_num*num_frames_per_clip,num_frames_per_clip):    #每16帧的开头
        frm_16or8=[]
        for i in range(s_index, s_index + num_frames_per_clip):
            #print(i,"**")
            image_name = str(filename) + '/' + str(filenames[i])
            img = Image.open(image_name)
            img_data = np.array(img)
            #ret_arr.append(img_data)
            frm_16or8.append(img_data)
        #print("done!!!")
        ret_arr.append(frm_16or8)
    return ret_arr, s_index

def read_clip_and_label(filename, batch_size, seq_num, start_pos=-1, num_frames_per_clip=16, crop_size=112, shuffle=False):
    lines = open(filename,'r')
    read_dirnames = []
    data = []
    label = []
    batch_index = 0
    next_batch_start = -1
    lines = list(lines)
    #print(lines)
    np_mean = np.load('crop_mean.npy').reshape([num_frames_per_clip, crop_size, crop_size, 3])
    # Forcing shuffle, if start_pos is not specified
    if start_pos < 0:
        shuffle = True
    if shuffle:
        #video_indices = range(len(lines))
        random.seed(time.time())
        video_indices= list(range(len(lines)))
        random.shuffle(video_indices)
    else:
    # Process videos sequentially
        video_indices = range(start_pos, len(lines))
        
    #print("*********************",video_indices)
    for index in video_indices:
        if(batch_index>=batch_size):
            next_batch_start = index
            break
        line = lines[index].strip('\n').split()
        dirname = line[0]
        tmp_label = line[1]
        #print("Loading a video clip from {}...".format(dirname))
        if not shuffle:
            print("Loading a video clip from {}...".format(dirname))
            
        tmp_data, _ = get_frames_data(dirname,seq_num, num_frames_per_clip)
        img_datas = [];
        if(len(tmp_data)!=0):
            for each_seqfrm in range(len(tmp_data)):
                seqfrm_data=[]
                for j in xrange(len(tmp_data[each_seqfrm])):
                    img = Image.fromarray(tmp_data[each_seqfrm][j].astype(np.uint8))
                    #print("********************",img)
                    if(img.width>img.height):
                        scale = float(crop_size)/float(img.height)
                        img = np.array(cv2.resize(np.array(img),(int(img.width * scale + 1), crop_size))).astype(np.float32)
                    else:
                        scale = float(crop_size)/float(img.width)
                        img = np.array(cv2.resize(np.array(img),(crop_size, int(img.height * scale + 1)))).astype(np.float32)                       
                    img = img[int((img.shape[0] - crop_size)/2):int((img.shape[0] - crop_size)/2 + crop_size), int((img.shape[1] - crop_size)/2):int((img.shape[1] - crop_size)/2 + crop_size),:] - np_mean[j]

                    seqfrm_data.append(img)

                img_datas.append(seqfrm_data)            
            data.append(img_datas)
            label.append(int(tmp_label))
            batch_index = batch_index + 1
            read_dirnames.append(dirname)

  # pad (duplicate) data/label if less than batch_size
    valid_len = len(data)
    #pad_len = batch_size - valid_len
    #if pad_len:
    #    for i in range(pad_len):
    #        data.append(img_datas)
    #        label.append(int(tmp_label))

    np_arr_data = np.array(data).astype(np.float32)
    np_arr_label = np.array(label).astype(np.int64)

    return np_arr_data, np_arr_label, next_batch_start, read_dirnames,valid_len
