#!/usr/bin/env python
# coding:utf-8

# 新建数据处理文件

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# import tensorflow as tf
import os
import numpy as np

# 获取文件路径和标签
def get_files(file_dir):
    cats = []
    label_cats = []
    dogs = []
    label_dogs = []
    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        if 'cat' in name[0]:
            cats.append(file_dir + file)
            label_cats.append(0)
        else:
            if 'dog' in name[0]:
                dogs.append(file_dir + file)
                label_dogs.append(1)
    # print('数据集中共有 %d 猫\n数据集中共有 %d 狗' %(len(cats), len(dogs)))



    # 打乱图片的顺序：
    image_list = np.hstack((cats, dogs))
    label_list = np.hstack((label_cats, label_dogs))
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    # 取出第一个元素作为 image 第二个元素作为 label
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]
    return image_list, label_list

# 创建一个batch：

# image_W ,image_H 指定图片的weight和height
# batch_size 每批读取的个数
# capacity队列中 最多容纳元素的个数
def get_batch(image, label, image_W, image_H, batch_size, capacity):
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    input_queue = tf.train.slice_input_producer([image, label])
    label = input_queue[1]

    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    image = tf.image.per_image_standardization(image)

    # num_threads 有多少个线程根据电脑配置设置
    # capacity 队列中 最多容纳图片的个数
    # tf.train.shuffle_batch 打乱顺序，
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,
                                              capacity=capacity)

    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch
