from data_manager import DataManager
from nn_classifier import NN
import numpy as np

import matplotlib.pyplot as plt

CLASS_LABELS = ['apple', 'banana', 'nectarine', 'plum', 'peach', 'watermelon',
                'pear', 'mango', 'grape', 'orange', 'strawberry', 'pineapple',
                'radish', 'carrot', 'potato', 'tomato', 'bellpepper', 'broccoli',
                'cabbage', 'cauliflower', 'celery', 'eggplant', 'garlic',
                'spinach', 'ginger']

image_size = 90
classes = CLASS_LABELS

dm = DataManager(classes, image_size)

val_data = dm.te_data
train_data = dm.tr_data

# X = np.empty((0, len(train_data[0, 2].flatten())))
X = train_data[:, 2]
arr = []
for i in range(train_data.shape[0]):
    arr.append(X[i].flatten())
    # X = np.vstack((X, train_data[i, 2].flatten()))

tr_data = {'X': np.asarray(arr), 'y': np.vstack(train_data[:, 1]).astype(np.float)}

# X = np.empty((0, len(val_data[0, 2].flatten())))
X = val_data[:, 2]
arr = []
for i in range(val_data.shape[0]):
    arr.append(X[i].flatten())
    # X = np.vstack((X, val_data[i, 2].flatten()))

te_data = {'X': np.asarray(arr), 'y': np.vstack(val_data[:, 1]).astype(np.float)}

K = [i for i in range(1, 101)]
test_losses = []
train_losses = []

for k in K:
    print(k)
    nn = NN(tr_data, te_data, n_neighbors=k)

    nn.train_model()
    print("trained")
    test_losses.append(nn.get_validation_error())
    train_losses.append(nn.get_train_error())

plt.plot(K, test_losses, label='Validation')
plt.plot(K, train_losses, label='Training')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Loss')
"""AWS"""
out_png = 'figures/3e.png'
plt.savefig(out_png, dpi=300)

# plt.show()
