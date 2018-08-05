# -*- encoding: utf8 -*-
# author: ronniecao
import sys
import os
import numpy
import matplotlib.pyplot as plt
import tensorflow as tf
from src.layer.conv_layer import ConvLayer
from src.layer.dense_layer import DenseLayer
from src.layer.pool_layer import PoolLayer


class ConvNet():

    def __init__(self, n_channel=3, n_classes=10, image_size=24):
        # 输入变量
        self.images = tf.placeholder(
            dtype=tf.float32, shape=[None, image_size, image_size, n_channel], name='images')
        self.labels = tf.placeholder(
            dtype=tf.int64, shape=[None], name='labels')
        self.keep_prob = tf.placeholder(
            dtype=tf.float32, name='keep_prob')
        self.global_step = tf.Variable(
            0, dtype=tf.int32, name='global_step')
        # print(self.images)
        # 网络结构
        conv_layer1 = ConvLayer(
            input_shape=(None, image_size, image_size, n_channel),
            n_size=3, n_filter=64, stride=1,
            activation='relu', batch_normal=True,
            weight_decay=1e-4, name='conv1')
        pool_layer1 = PoolLayer(
            n_size=2, stride=2, mode='max', resp_normal=True, name='pool1')

        conv_layer2 = ConvLayer(
            input_shape=(None, int(image_size/2), int(image_size/2), 64), n_size=3, n_filter=128,
            stride=1, activation='relu', batch_normal=True, weight_decay=1e-4,
            name='conv2')
        pool_layer2 = PoolLayer(
            n_size=2, stride=2, mode='max', resp_normal=True, name='pool2')

        conv_layer3 = ConvLayer(
            input_shape=(None, int(image_size/4), int(image_size/4), 128), n_size=3, n_filter=256,
            stride=1, activation='relu', batch_normal=True, weight_decay=1e-4,
            name='conv3')
        pool_layer3 = PoolLayer(
            n_size=2, stride=2, mode='max', resp_normal=True, name='pool3')

        dense_layer1 = DenseLayer(
            input_shape=(None, int(image_size/8) * int(image_size/8) * 256), hidden_dim=1024,
            activation='relu', dropout=True, keep_prob=self.keep_prob,
            batch_normal=True, weight_decay=1e-4, name='dense1')

        dense_layer2 = DenseLayer(
            input_shape=(None, 1024), hidden_dim=n_classes,
            activation='none', dropout=False, keep_prob=None,
            batch_normal=False, weight_decay=1e-4, name='dense2')

        # 数据流
        hidden_conv1 = conv_layer1.get_output(input=self.images)
        hidden_pool1 = pool_layer1.get_output(input=hidden_conv1)
        hidden_conv2 = conv_layer2.get_output(input=hidden_pool1)
        hidden_pool2 = pool_layer2.get_output(input=hidden_conv2)
        hidden_conv3 = conv_layer3.get_output(input=hidden_pool2)
        hidden_pool3 = pool_layer3.get_output(input=hidden_conv3)
        input_dense1 = tf.reshape(
            hidden_pool3, [-1, int(image_size/8) * int(image_size/8) * 256]
        )
        # print("input dense1:", input_dense1)
        output_dense1 = dense_layer1.get_output(input=input_dense1)
        # print("output ", output_dense1)
        logits = dense_layer2.get_output(input=output_dense1)
        print("logits: ", logits)
        # 目标函数
        self.objective = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=self.labels))
        tf.add_to_collection('losses', self.objective)
        self.avg_loss = tf.add_n(tf.get_collection('losses'))
        # 优化器
        lr = tf.cond(tf.less(self.global_step, 50000),
                     lambda: tf.constant(0.01),
                     lambda: tf.cond(tf.less(self.global_step, 100000),
                                     lambda: tf.constant(0.001),
                                     lambda: tf.constant(0.0001)))
        # tf.summary.scalar('learning rate', lr)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(
            self.avg_loss, global_step=self.global_step)

        # 观察值
        correct_prediction = tf.equal(self.labels, tf.argmax(logits, 1))
        # tf.summary.scalar('correct prediction', correct_prediction)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        # tf.summary.scalar('accuracy', self.accuracy)

    def train(self, dataloader, backup_path, n_epoch=5, batch_size=128):
        # 构建会话
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # 模型保存器
        self.saver = tf.train.Saver(
            var_list=tf.global_variables(), write_version=tf.train.SaverDef.V2,
            max_to_keep=10)
        # 模型初始化
        # 这个是tensorboard的初始化
        train_writer = tf.summary.FileWriter(
            'basic_cnn/train_writer', self.sess.graph
        )
        tf.summary.scalar('loss', self.avg_loss)
        tf.summary.scalar('accuracy', self.accuracy)
        self.merged = tf.summary.merge_all()
        self.sess.run(tf.global_variables_initializer())
        # 模型训练
        for epoch in range(0, n_epoch+1):
            # 数据增强
            train_images = dataloader.data_augmentation(
                dataloader.train_images, mode='train',
                flip=True, crop=True,
                crop_shape=(24, 24, 3),
                whiten=True, noise=False)
            train_labels = dataloader.train_labels
            valid_images = dataloader.data_augmentation(
                dataloader.valid_images, mode='test',
                flip=False, crop=True,
                crop_shape=(24, 24, 3),
                whiten=True, noise=False)
            valid_labels = dataloader.valid_labels

            # 开始本轮的训练，并计算目标函数值
            train_loss = 0.0
            for i in range(0, dataloader.n_train, batch_size):
                batch_images = train_images[i: i+batch_size]
                batch_labels = train_labels[i: i+batch_size]

                [_, avg_loss, iteration] = self.sess.run(
                    fetches=[self.optimizer, self.avg_loss, self.global_step],
                    feed_dict={self.images: batch_images,
                               self.labels: batch_labels,
                               self.keep_prob: 0.5})
                # print(summary)
                # file_writer.add_summary(summary, global_step=self.global_step)
                train_loss += avg_loss * batch_images.shape[0]
            train_loss = 1.0 * train_loss / dataloader.n_train

            # 在训练之后，获得本轮的验证集损失值和准确率
            valid_accuracy, valid_loss = 0.0, 0.0
            for i in range(0, dataloader.n_valid, batch_size):
                batch_images = valid_images[i: i+batch_size]
                batch_labels = valid_labels[i: i+batch_size]
                [avg_accuracy, avg_loss] = self.sess.run(
                    fetches=[self.accuracy, self.avg_loss],
                    feed_dict={
                        self.images: batch_images,
                        self.labels: batch_labels,
                        self.keep_prob: 1.0})
                # print([avg_accuracy, avg_loss])
                summary, _ = self.sess.run(
                    [self.merged, self.global_step],
                    feed_dict={
                        self.images: batch_images,
                        self.labels: batch_labels,
                        self.keep_prob: 1.0
                    }
                )
                train_writer.add_summary(
                    summary, ((epoch*dataloader.n_valid)/batch_size)*batch_size+i)
                # print("[+]", summary)
                valid_accuracy += avg_accuracy * batch_images.shape[0]
                valid_loss += avg_loss * batch_images.shape[0]
            valid_accuracy = 1.0 * valid_accuracy / dataloader.n_valid
            valid_loss = 1.0 * valid_loss / dataloader.n_valid

            print('epoch{%d}, iter[%d], train loss: %.6f, '
                  'valid precision: %.6f, valid loss: %.6f' % (
                      epoch, iteration, train_loss, valid_accuracy, valid_loss))
            sys.stdout.flush()

            # 保存模型
            if epoch <= 1000 and epoch % 100 == 0 or \
                    epoch <= 10000 and epoch % 1000 == 0:
                saver_path = self.saver.save(
                    self.sess, os.path.join(backup_path, 'model_%d.ckpt' % (epoch)))

        self.sess.close()

    def test(self, dataloader, backup_path, epoch, batch_size=128):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # 读取模型
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        model_path = os.path.join(backup_path, 'model_%d.ckpt' % (epoch))
        assert(os.path.exists(model_path+'.index'))
        self.saver.restore(self.sess, model_path)
        print('[+] read model from %s' % (model_path))
        # 在测试集上计算准确率
        accuracy_list = []
        # print(type(dataloader.test_images))
        # ttt=0
        # import cv2
        # while True:
        #     cv2.imshow('pic', dataloader.test_images[ttt])
        #     cv2.waitKey(0)
        #     ttt=ttt+1
        #     cv2.destroyAllWindows()
        test_images = dataloader.data_augmentation(
            dataloader.test_images,
            flip=False, crop=True,
            crop_shape=(24, 24, 3),
            whiten=True, noise=False)
        # ttt=0
        # import cv2
        # while True:
        #     cv2.imshow('pic', dataloader.test_images[ttt])
        #     cv2.waitKey(0)
        #     ttt=ttt+1
        #     cv2.destroyAllWindows()
        test_labels = dataloader.test_labels
        for i in range(0, dataloader.n_test, batch_size):
            batch_images = test_images[i: i+batch_size]
            batch_labels = test_labels[i: i+batch_size]
            # print(batch_labels)
            [avg_accuracy] = self.sess.run(
                fetches=[self.accuracy],
                feed_dict={
                    self.images: batch_images,
                    self.labels: batch_labels,
                    self.keep_prob: 1.0
                }
            )
            # print([avg_accuracy])
            # input()
            accuracy_list.append(avg_accuracy)
        print('test precision: %.4f' % (numpy.mean(accuracy_list)))
        self.sess.close()

    def debug(self):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        [temp] = sess.run(
            fetches=[self.observe],
            feed_dict={self.images: numpy.random.random(size=[128, 24, 24, 3]),
                       self.labels: numpy.random.randint(low=0, high=9, size=[128, ]),
                       self.keep_prob: 1.0})
        print(temp)
