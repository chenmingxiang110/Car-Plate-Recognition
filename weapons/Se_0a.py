import numpy as np
import tensorflow as tf

class seg_model:
    """
    The model is based on deeplabv3 while the net structure is extremely shrinked
    using resnet-12 with four pyramid layers.
    """

    def __init__(self):
        self.xs = tf.placeholder(tf.float32, [None, 512, 512, 3])
        self.ys = tf.placeholder(tf.float32, [None, 64, 64])
        self.learning_rate = tf.placeholder(tf.float32)
        self.isTrain = tf.placeholder(tf.bool, name='phase')
        self.weight_decay = tf.placeholder(tf.float32)
        self.bn_momentum = tf.placeholder(tf.float32)

        with tf.variable_scope("block1"):
            conv11 = self._nn_conv_bn_layer(self.xs, 'conv_11', [7, 7, 3, 64], [2, 2])
            conv12 = self._nn_conv_bn_layer(conv11, 'conv_12', [5, 5, 64, 64], [1, 1])
        with tf.variable_scope("block2"):
            conv21 = self._nn_conv_bn_layer(conv12, 'conv_21', [3, 3, 64, 64], [1, 1])
            conv22 = self._nn_conv_bn_layer(conv21, 'conv_22', [3, 3, 64, 64], [1, 1])
            conv22 = tf.add(conv12, conv22)
            conv22_shortcut = self._nn_conv_bn_layer(conv22, 'conv_24s', [3, 3, 64, 128], [2, 2])
        with tf.variable_scope("block3"):
            conv31 = self._nn_conv_bn_layer(conv22, 'conv_31', [3, 3, 64, 128], [2, 2])
            conv32 = self._nn_conv_bn_layer(conv31, 'conv_32', [3, 3, 128, 128], [1, 1])
            conv32 = tf.add(conv22_shortcut, conv32)
            conv33 = self._nn_conv_bn_layer(conv32, 'conv_33', [3, 3, 128, 128], [1, 1])
            conv34 = self._nn_conv_bn_layer(conv33, 'conv_34', [3, 3, 128, 128], [1, 1])
            conv34 = tf.add(conv32, conv34)
            conv34_shortcut = self._nn_conv_bn_layer(conv34, 'conv_34s', [3, 3, 128, 256], [2, 2])
        with tf.variable_scope("block4"):
            conv41 = self._nn_conv_bn_layer(conv34, 'conv_41', [3, 3, 128, 256], [2, 2])
            conv42 = self._nn_conv_bn_layer(conv41, 'conv_42', [3, 3, 256, 256], [1, 1])
            conv42 = tf.add(conv34_shortcut, conv42)
        with tf.variable_scope("block5"):
            conv5 = self._nn_atrous_conv_bn_layer(conv42, 'conv_5', [3, 3, 256, 512], 2)
            conv6 = self._nn_atrous_conv_bn_layer(conv5, 'conv_6', [3, 3, 512, 512], 2)
        with tf.variable_scope("block_pyramid"):
            pyramid1 = self._nn_conv_bn_layer(conv6, 'pyramid1', [1, 1, 512, 128], [1, 1])
            pyramid2 = self._nn_atrous_conv_bn_layer(conv6, 'pyramid2', [3, 3, 512, 128], 6)
            pyramid3 = self._nn_atrous_conv_bn_layer(conv6, 'pyramid3', [3, 3, 512, 128], 12)
            pyramid4 = self._nn_atrous_conv_bn_layer(conv6, 'pyramid4', [3, 3, 512, 128], 18)
            pyramid_concat = tf.concat([pyramid1, pyramid2, pyramid3, pyramid4], 3, name='pyramid_concat')
            conv_final = self._nn_conv_bn_layer(pyramid_concat, 'conv_final', [1, 1, 512, 1], [1, 1], activition = None)
            conv_reshape = tf.reshape(conv_final, [-1, 64, 64], name = 'conv_reshape')
            self.pred = tf.nn.sigmoid(conv_reshape)
            # bool_pred = tf.greater(conv_reshape, 0)
            # self.pred = tf.cast(bool_pred, tf.float32)

        # self.loss = tf.add(tf.reduce_mean(tf.square(self.pred-self.ys)), tf.add_n(tf.get_collection('loss')))
        self.loss = tf.reduce_mean(tf.square(self.pred-self.ys))
        optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1 = 0.9, beta2 = 0.99)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            gvs = optimizer.compute_gradients(self.loss)
            capped_gvs = [(tf.clip_by_value(grad, -400., 400.), var) for grad, var in gvs if grad is not None]
            self.train_op = optimizer.apply_gradients(capped_gvs)

        self.loss_summary = tf.summary.scalar("loss", self.loss)
        self.merged = tf.summary.merge_all()

    def _nn_conv_bn_layer(self, inputs, scope, shape, strides, activition = tf.nn.relu6):
        with tf.variable_scope(scope):
            W_conv = tf.get_variable("W", shape=shape, initializer=tf.contrib.layers.xavier_initializer())
            h_conv = tf.nn.conv2d(inputs, W_conv, strides=[1, strides[0], strides[1], 1],
                                  padding='SAME', name="conv2d")
            # tf.add_to_collection("loss", tf.reduce_sum(self.weight_decay*tf.square(W_conv)))
            b = tf.get_variable("bias" , shape=[shape[3]], initializer=tf.contrib.layers.xavier_initializer())
            if activition is not None:
                h_bn = tf.layers.batch_normalization(h_conv+b, training = self.isTrain, momentum = self.bn_momentum)
                h_relu = activition(h_bn, name="activition")
                return h_relu
            return h_conv+b

    def _nn_atrous_conv_bn_layer(self, inputs, scope, shape, rate, activition = tf.nn.relu6):
        with tf.variable_scope(scope):
            W_conv = tf.get_variable("W", shape=shape, initializer=tf.contrib.layers.xavier_initializer())
            h_conv = tf.nn.atrous_conv2d(inputs, W_conv, rate,
                                         padding='SAME', name="atrous_conv2d")
            # tf.add_to_collection("loss", tf.reduce_sum(self.weight_decay*tf.square(W_conv)))
            b = tf.get_variable("bias" , shape=[shape[3]], initializer=tf.contrib.layers.xavier_initializer())
            if activition is not None:
                h_bn = tf.layers.batch_normalization(h_conv+b, training = self.isTrain, momentum = self.bn_momentum)
                h_relu = activition(h_bn, name="activition")
                return h_relu
            return h_conv+b

    def train(self, sess, learning_rate, bn_momentum, xs, ys, weight_decay = 1e-4):
        _, loss, pred, summary = sess.run([self.train_op, self.loss, self.pred, self.merged],
                                           feed_dict = {self.isTrain: True,
                                                        self.weight_decay: weight_decay,
                                                        self.bn_momentum: bn_momentum,
                                                        self.learning_rate: learning_rate,
                                                        self.xs: xs, self.ys: ys})
        return loss, pred, summary

    def get_loss(self, sess, xs, ys, weight_decay = 1e-4):
        loss = sess.run(self.loss, feed_dict = {self.isTrain: False,
                                                self.weight_decay: weight_decay,
                                                self.xs: xs, self.ys: ys})
        return loss

    def predict(self, sess, xs):
        prediction = sess.run(self.pred, feed_dict = {self.isTrain: False,
                                                      self.xs: xs})
        return prediction
