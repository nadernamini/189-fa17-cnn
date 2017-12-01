from numpy.random import uniform
import random
import time

import numpy as np
import glob
import os

import matplotlib.pyplot as plt
import sys

from sklearn.neighbors import KNeighborsClassifier


class NN:
    def __init__(self, train_data, val_data, n_neighbors=5):
        X = np.empty((0, len(train_data[0, 2].flatten())))

        for i in range(train_data.shape[0]):
            print(i)
            X = np.vstack((X, train_data[i, 2].flatten()))

        self.train_data = {'X': X, 'y': np.vstack(train_data[:, 1]).astype(np.float)}
        # print(train_data[:, 1], type(train_data[:, 1]))
        X = np.empty((0, len(val_data[0, 2].flatten())))
        for i in range(val_data.shape[0]):
            print(i)
            X = np.vstack((X, val_data[i, 2].flatten()))

        self.val_data = {'X': X, 'y': np.vstack(val_data[:, 1]).astype(np.float)}

        print(val_data.shape, train_data.shape)

        self.sample_size = 400

        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def train_model(self):
        """
        Train Nearest Neighbors model
        """
        # print(self.train_data['X'], self.train_data['y'].shape)
        X, y = self.train_data['X'], self.train_data['y']

        self.model.fit(X, y)
        return self.model

    def get_validation_error(self):
        """
        Compute validation error. Please only compute the error on the sample_size number
        over randomly selected data points. To save computation.
        """
        chc = np.random.randint(self.val_data['X'].shape[0], size=self.sample_size)
        X, y = self.val_data['X'][chc], self.val_data['y'][chc]

        yhat = self.model.predict(X)
        # self.model.score(self.val_data['X'][chc], self.val_data['y'][chc])

        return self.calc_err(y, yhat)

    def calc_err(self, y, yhat):
        err = 0
        for i in range(self.sample_size):
            err += np.linalg.norm(y[i] - yhat[i]) ** 2
        return err / self.sample_size

    def get_train_error(self):
        """
        Compute train error. Please only compute the error on the sample_size number
        over randomly selected data points. To save computation.
        """
        chc = np.random.randint(self.train_data['X'].shape[0], size=self.sample_size)
        X, y = self.train_data['X'][chc], self.train_data['y'][chc]
        yhat = self.model.predict(X)
        # self.model.score(self.train_data['X'][chc], self.train_data['y'][chc])

        return self.calc_err(y, yhat)
