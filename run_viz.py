import pickle
from viz_features import Viz_Feat
from train_cnn import val_data, train_data, CLASS_LABELS

with open(r"solver.pickle", "rb") as input_file:
    solver = pickle.load(input_file)

with open(r"cnn.pickle", "rb") as input_file:
    cnn = pickle.load(input_file)


sess = solver.sess

cm = Viz_Feat(val_data, train_data, CLASS_LABELS, sess)

cm.vizualize_features(cnn)
