import tensorflow as tf
import datetime
import os
import sys
import argparse
from data_manager import DataManager

slim = tf.contrib.slim


class Solver(object):
    def __init__(self, net, data):
        self.net = net
        self.data = data

        # Number of iterations to train for
        self.max_iter = 5000
        # Every 200 iterations record the test and train loss
        self.summary_iter = 200
        self.train_accuracy = None
        self.test_accuracy = None
        """
        Tensorflow is told to use a gradient descent optimizer 
        In the function optimize you will iteratively apply this on batches of data
        """
        self.train_step = tf.train.MomentumOptimizer(.003, .9)
        self.train = self.train_step.minimize(self.net.class_loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def optimize(self):
        # Append the current train and test accuracy every 200 iterations
        self.train_accuracy = []
        self.test_accuracy = []

        """
        Performs the training of the network. 
        Implement SGD using the data manager to compute the batches
        Make sure to record the training and test accuracy through out the process
        """
        for i in range(self.max_iter):
            print("iteration " + str(i))
            images, labels = self.data.get_train_batch()
            # print(images, labels)
            feed_dict_train = {self.net.images: images, self.net.labels: labels}
            self.sess.run([self.train], feed_dict=feed_dict_train)
            if i % self.summary_iter == 0:
                train_loss = self.sess.run(self.net.accurracy, feed_dict=feed_dict_train)
                self.train_accuracy.append(train_loss)
                images, labels = self.data.get_validation_batch()
                feed_dict_test = {self.net.images: images, self.net.labels: labels}
                test_loss = self.sess.run(self.net.accurracy, feed_dict=feed_dict_test)
                self.test_accuracy.append(test_loss)


