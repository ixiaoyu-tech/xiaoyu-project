# coding=utf-8
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
# noinspection PyUnresolvedReferences
import input_data
import numpy as np
import model
import os


def get_one_image(train):
    files = os.listdir(train)
    n = len(files)
    ind = np.random.randint(0, n)
    img_dir = os.path.join(train, files[ind])
    image = Image.open(img_dir)
    plt.imshow(image)
    plt.show()
    image = image.resize([208, 208])
    image = np.array(image)
    return image


def evaluate_one_image():
    train = '/Users/fq/Documents/my_study/python_code/Cats_vs_Dogs/data/test/'

    image_array = get_one_image(train)

    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 2
        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)


        image = tf.reshape(image, [1, 208, 208, 3])
        logit = model.inference(image, BATCH_SIZE, N_CLASSES)

        logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32, shape=[208, 208, 3])

        # 存放模型
        logs_train_dir = '/Users/fq/Documents/my_study/python_code/Cats_vs_Dogs/logs'
        saver = tf.train.Saver()

        with tf.Session() as sess:

            print("从指定的路径中加载模型......")
            # 将模型加载到sess 中
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('模型加载成功, 训练的步数为 %s' % global_step)
            else:
                print('模型加载失败，，，文件没有找到')
                # 将图片输入到模型计算
            prediction = sess.run(logit, feed_dict={x: image_array})
            # 获取输出结果中最大概率的索引
            max_index = np.argmax(prediction)
        if max_index == 0:
            print('狗的概率 %.6f' % prediction[:, 0])
        else:
            print('狗的概率 %.6f' % prediction[:, 1])

evaluate_one_image()


