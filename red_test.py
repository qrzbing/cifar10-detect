# -*- encoding: utf8 -*-
# author: ronniecao
import sys
import os
import cv2
import numpy as np
import numpy
import matplotlib.pyplot as plt
import tensorflow as tf
from src.layer.conv_layer import ConvLayer
from src.layer.dense_layer import DenseLayer
from src.layer.pool_layer import PoolLayer
from src.data.cifar10 import Corpus


class ConvNet():

    def __init__(self, n_channel=3, n_classes=10, image_size=24, n_layers=20):
        # 设置超参数
        self.n_channel = n_channel
        self.n_classes = n_classes
        self.image_size = image_size
        self.n_layers = n_layers

        # 输入变量
        self.images = tf.placeholder(
            dtype=tf.float32, shape=[
                None, self.image_size, self.image_size, self.n_channel],
            name='images')
        self.labels = tf.placeholder(
            dtype=tf.int64, shape=[None], name='labels')
        self.keep_prob = tf.placeholder(
            dtype=tf.float32, name='keep_prob')
        self.global_step = tf.Variable(
            0, dtype=tf.int32, name='global_step')

        # 网络输出
        self.logits = self.inference(self.images)

        # 目标函数
        self.objective = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.labels)
        )
        tf.add_to_collection('losses', self.objective)
        self.avg_loss = tf.add_n(tf.get_collection('losses'))
        # 优化器
        lr = tf.cond(
            tf.less(self.global_step, 50000),
            lambda: tf.constant(0.01),
            lambda: tf.cond(
                tf.less(self.global_step, 100000),
                lambda: tf.constant(0.005),
                lambda: tf.cond(
                    tf.less(self.global_step, 150000),
                    lambda: tf.constant(
                        0.0025),
                    lambda: tf.constant(0.001)
                )
            )
        )
        # tf.summary.scalar('learning rate', lr)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(
            self.avg_loss, global_step=self.global_step)

        # 观察值
        correct_prediction = tf.equal(self.labels, tf.argmax(self.logits, 1))
        # tf.summary.scalar('correct prediction', correct_prediction)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        # tf.summary.scalar('accuracy', self.accuracy)

    def inference(self, images):  # 前向传播
        n_layers = int((self.n_layers - 2) / 6)
        # 网络结构
        # 第一层卷积
        conv_layer0_list = []
        conv_layer0_list.append(
            ConvLayer(
                input_shape=(None, self.image_size,
                             self.image_size, self.n_channel),
                n_size=3, n_filter=64, stride=1, activation='relu',
                batch_normal=True, weight_decay=1e-4, name='conv0'))

        conv_layer1_list = []
        for i in range(1, n_layers+1):
            conv_layer1_list.append(
                ConvLayer(
                    input_shape=(None, self.image_size, self.image_size, 64),
                    n_size=3, n_filter=64, stride=1, activation='relu',
                    batch_normal=True, weight_decay=1e-4, name='conv1_%d' % (2*i-1)))
            conv_layer1_list.append(
                ConvLayer(
                    input_shape=(None, self.image_size, self.image_size, 64),
                    n_size=3, n_filter=64, stride=1, activation='none',
                    batch_normal=True, weight_decay=1e-4, name='conv1_%d' % (2*i)))

        conv_layer2_list = []
        conv_layer2_list.append(
            ConvLayer(
                input_shape=(None, self.image_size, self.image_size, 64),
                n_size=3, n_filter=128, stride=2, activation='relu',
                batch_normal=True, weight_decay=1e-4, name='conv2_1'))
        conv_layer2_list.append(
            ConvLayer(
                input_shape=(None, int(self.image_size)/2,
                             int(self.image_size)/2, 128),
                n_size=3, n_filter=128, stride=1, activation='none',
                batch_normal=True, weight_decay=1e-4, name='conv2_2'))
        for i in range(2, n_layers+1):
            conv_layer2_list.append(
                ConvLayer(
                    input_shape=(None, int(self.image_size/2),
                                 int(self.image_size/2), 128),
                    n_size=3, n_filter=128, stride=1, activation='relu',
                    batch_normal=True, weight_decay=1e-4, name='conv2_%d' % (2*i-1)))
            conv_layer2_list.append(
                ConvLayer(
                    input_shape=(None, int(self.image_size/2),
                                 int(self.image_size/2), 128),
                    n_size=3, n_filter=128, stride=1, activation='none',
                    batch_normal=True, weight_decay=1e-4, name='conv2_%d' % (2*i)))

        conv_layer3_list = []
        conv_layer3_list.append(
            ConvLayer(
                input_shape=(None, int(self.image_size/2),
                             int(self.image_size/2), 128),
                n_size=3, n_filter=256, stride=2, activation='relu',
                batch_normal=True, weight_decay=1e-4, name='conv3_1'))
        conv_layer3_list.append(
            ConvLayer(
                input_shape=(None, int(self.image_size/4),
                             int(self.image_size/4), 256),
                n_size=3, n_filter=256, stride=1, activation='relu',
                batch_normal=True, weight_decay=1e-4, name='conv3_2'))
        for i in range(2, n_layers+1):
            conv_layer3_list.append(
                ConvLayer(
                    input_shape=(None, int(self.image_size/4),
                                 int(self.image_size/4), 256),
                    n_size=3, n_filter=256, stride=1, activation='relu',
                    batch_normal=True, weight_decay=1e-4, name='conv3_%d' % (2*i-1)))
            conv_layer3_list.append(
                ConvLayer(
                    input_shape=(None, int(self.image_size/4),
                                 int(self.image_size/4), 256),
                    n_size=3, n_filter=256, stride=1, activation='none',
                    batch_normal=True, weight_decay=1e-4, name='conv3_%d' % (2*i)))

        dense_layer1 = DenseLayer(
            input_shape=(None, 256),
            hidden_dim=self.n_classes,
            activation='none', dropout=False, keep_prob=None,
            batch_normal=False, weight_decay=1e-4, name='dense1')

        # 数据流
        hidden_conv = conv_layer0_list[0].get_output(input=images)

        for i in range(0, n_layers):
            hidden_conv1 = conv_layer1_list[2*i].get_output(input=hidden_conv)
            hidden_conv2 = conv_layer1_list[2 *
                                            i+1].get_output(input=hidden_conv1)
            hidden_conv = tf.nn.relu(hidden_conv + hidden_conv2)

        hidden_conv1 = conv_layer2_list[0].get_output(input=hidden_conv)
        hidden_conv2 = conv_layer2_list[1].get_output(input=hidden_conv1)
        hidden_pool = tf.nn.max_pool(
            hidden_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        hidden_pad = tf.pad(hidden_pool, [[0, 0], [0, 0], [0, 0], [32, 32]])
        hidden_conv = tf.nn.relu(hidden_pad + hidden_conv2)
        for i in range(1, n_layers):
            hidden_conv1 = conv_layer2_list[2*i].get_output(input=hidden_conv)
            hidden_conv2 = conv_layer2_list[2 *
                                            i+1].get_output(input=hidden_conv1)
            hidden_conv = tf.nn.relu(hidden_conv + hidden_conv2)

        hidden_conv1 = conv_layer3_list[0].get_output(input=hidden_conv)
        hidden_conv2 = conv_layer3_list[1].get_output(input=hidden_conv1)
        hidden_pool = tf.nn.max_pool(
            hidden_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        hidden_pad = tf.pad(hidden_pool, [[0, 0], [0, 0], [0, 0], [64, 64]])
        hidden_conv = tf.nn.relu(hidden_pad + hidden_conv2)
        for i in range(1, n_layers):
            hidden_conv1 = conv_layer3_list[2*i].get_output(input=hidden_conv)
            hidden_conv2 = conv_layer3_list[2 *
                                            i+1].get_output(input=hidden_conv1)
            hidden_conv = tf.nn.relu(hidden_conv + hidden_conv2)

        # global average pooling
        input_dense1 = tf.reduce_mean(hidden_conv, reduction_indices=[1, 2])
        logits = dense_layer1.get_output(input=input_dense1)

        return logits

    def test(self, backup_path='backup/cifar10-v16/', epoch=500, image_str=''):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # 读取模型
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        model_path = os.path.join(backup_path, 'model_%d.ckpt' % (epoch))
        assert(os.path.exists(model_path+'.index'))
        self.saver.restore(self.sess, model_path)
        print('read model from %s' % (model_path))
        image = cv2.imread(image_str)
        image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_CUBIC)
        image = image.astype(float)
        images = []
        images.append(image)
        images = np.array(images, dtype='float')
        cifar10 = Corpus()
        test_images = cifar10.data_augmentation(
            images,
            flip=False,
            crop=True,
            crop_shape=(24, 24, 3),
            whiten=True,
            noise=False
        )
        test_labels = np.array(range(10))
        tst_list = []
        for i in range(10):
            # print("[+] ", test_labels[i])
            [avg_accuracy] = self.sess.run(
                fetches=[self.accuracy],
                feed_dict={
                    self.images: test_images,
                    self.labels: [test_labels[i]],
                    self.keep_prob: 1.0
                }
            )
            # print(avg_accuracy)
            tst_list.append(avg_accuracy)
            # print(image_str)
        # print(tst_list)
        self.sess.close()
        return tst_list

