import pickle
from viz_features import Viz_Feat


CLASS_LABELS = ['apple', 'banana', 'nectarine', 'plum', 'peach', 'watermelon', 'pear', 'mango', 'grape', 'orange',
                'strawberry', 'pineapple',
                'radish', 'carrot', 'potato', 'tomato', 'bellpepper', 'broccoli', 'cabbage', 'cauliflower', 'celery',
                'eggplant', 'garlic', 'spinach', 'ginger']

LITTLE_CLASS_LABELS = ['apple', 'banana', 'eggplant']

with open(r"solver.pickle", "rb") as input_file:
    solver = pickle.load(input_file)

with open(r"cnn.pickle", "rb") as input_file:
    cnn = pickle.load(input_file)

with open(r"val.pickle", "rb") as input_file:
    val_data = pickle.load(input_file)

with open(r"train.pickle", "rb") as input_file:
    train_data = pickle.load(input_file)


sess = solver.sess

cm = Viz_Feat(val_data, train_data, CLASS_LABELS, sess)

cm.vizualize_features(cnn)
