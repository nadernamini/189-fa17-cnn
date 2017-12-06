import matplotlib.pyplot as plt
import numpy as np
import cv2


class Viz_Feat(object):
    def __init__(self, val_data, train_data, class_labels, sess):
        self.val_data = val_data
        self.train_data = train_data
        self.CLASS_LABELS = class_labels
        self.sess = sess

    def vizualize_features(self, net):
        images = [0, 10, 100]
        """
        Compute the response map for the index images
        """
        for i in images:
            out = self.sess.run([net.conv_layer],
                                feed_dict={net.images: self.val_data[i, 2].reshape((1, 90, 90, 3)),
                                           net.labels: self.val_data[i, 1].reshape((1, 25))})
            for j in range(5):
                img = self.revert_image(out[0][0, :, :, j])
                plt.title('img-' + str(i) + '-filter-' + str(j+1))
                cv2.imwrite('figures/img-' + str(i) + '-filter-' + str(j+1) + '.png', img)

    def revert_image(self, img):
        """
        Used to revert images back to a form that can be easily visualized
        """

        img = (img + 1.0) / 2.0 * 255.0

        img = np.array(img, dtype=int)

        blank_img = np.zeros([img.shape[0], img.shape[1], 3])

        blank_img[:, :, 0] = img
        blank_img[:, :, 1] = img
        blank_img[:, :, 2] = img

        img = blank_img.astype("uint8")

        return img
