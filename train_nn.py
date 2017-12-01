from data_manager import DataManager
from nn_classifier import NN
from trainer import Solver

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

K = [1, 20, 100]
test_losses = []
train_losses = []

for k in K:
    nn = NN(train_data, val_data, n_neighbors=k)

    nn.train_model()

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

#####Plot the test error and training error###
