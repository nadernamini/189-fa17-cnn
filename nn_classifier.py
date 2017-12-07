import numpy as np

from sklearn.neighbors import KNeighborsClassifier


class NN:
    def __init__(self, train_data, val_data, n_neighbors=5):
        self.train_data = train_data

        self.val_data = val_data

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
        # err = 0
        # for i in range(self.sample_size):
        #     err += np.linalg.norm(y[i] - yhat[i]) ** 2
        return 1 - (np.linalg.norm(y - yhat) / (self.sample_size))


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
