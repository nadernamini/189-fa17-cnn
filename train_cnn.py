from data_manager import DataManager
from cnn import CNN
from trainer import Solver
import pickle
import random

import matplotlib.pyplot as plt

CLASS_LABELS = ['apple', 'banana', 'nectarine', 'plum', 'peach', 'watermelon', 'pear', 'mango', 'grape', 'orange',
                'strawberry', 'pineapple',
                'radish', 'carrot', 'potato', 'tomato', 'bellpepper', 'broccoli', 'cabbage', 'cauliflower', 'celery',
                'eggplant', 'garlic', 'spinach', 'ginger']

LITTLE_CLASS_LABELS = ['apple', 'banana', 'eggplant']

image_size = 90

random.seed(0)

classes = CLASS_LABELS
dm = DataManager(classes, image_size)

cnn = CNN(classes, image_size)

solver = Solver(cnn, dm)

solver.optimize()

plt.plot(solver.test_accuracy, label='Validation')
plt.plot(solver.train_accuracy, label='Training')
plt.legend()
plt.xlabel('Iterations (in 200s)')
plt.ylabel('Accuracy')
"""AWS"""
out_png = 'figures/3e.png'
plt.savefig(out_png, dpi=300)

# plt.show()

val_data = dm.val_data
train_data = dm.train_data

with open(r"pk/solver.pickle", "wb") as output_file:
    pickle.dump(solver, output_file)

with open(r"pk/cnn.pickle", "wb") as output_file:
    pickle.dump(cnn, output_file)

with open(r"pk/val.pickle", "wb") as output_file:
    pickle.dump(val_data, output_file)

with open(r"pk/train.pickle", "wb") as output_file:
    pickle.dump(train_data, output_file)

# sess = solver.sess



# cm = Viz_Feat(val_data, train_data, CLASS_LABELS, sess)
#
# cm.vizualize_features(cnn)
