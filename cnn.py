import tensorflow as tf

slim = tf.contrib.slim


class CNN(object):
    def __init__(self, classes, image_size):
        """
        Initializes the size of the network
        """

        self.classes = classes
        self.num_class = len(self.classes)
        self.image_size = image_size
        self.conv_layer = None

        self.output_size = self.num_class
        self.batch_size = 40

        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name='images')

        self.logits = self.build_network(self.images, num_outputs=self.output_size)

        self.labels = tf.placeholder(tf.float32, [None, self.num_class])

        self.loss_layer(self.logits, self.labels)
        self.total_loss = tf.losses.get_total_loss()
        tf.summary.scalar('total_loss', self.total_loss)

    def build_network(self,
                      images,
                      num_outputs,
                      scope='yolo'):
        with tf.variable_scope(scope):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                weights_regularizer=slim.l2_regularizer(0.0005)):
                '''
                Fill in network architecture here
                Network should start out with the images function
                Then it should return net
                '''
                net = slim.conv2d(images, 5, [15, 15], scope='conv_1')
                self.conv_layer = net
                net = slim.max_pool2d(net, [3, 3], scope='pool')
                net = slim.flatten(net, scope='flat_1')
                net = slim.fully_connected(net, 512, scope='fc_2')

                net = slim.fully_connected(net, self.output_size, scope='fc_3')
        return net

    @staticmethod
    def get_acc(y_, y_out):
        """
        Computes accuracy given two tensorflows arrays
        y_ (the true label) and y_out (the predict label)
        """

        cp = tf.equal(tf.argmax(y_out, 1), tf.argmax(y_, 1))
        ac = tf.reduce_mean(tf.cast(cp, tf.float32))

        return ac

    def loss_layer(self, predicts, classes, scope='loss_layer'):
        """
        The loss layer of the network, which is written for you.
        """
        with tf.variable_scope(scope):
            self.class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=classes, logits=predicts))
            self.accurracy = self.get_acc(classes, predicts)
