# -*- coding:utf-8 -*-
import lib.mlnet
import tensorflow as tf

class MLNet(lib.mlnet.MLNet):
    """
    Metric Learning Net restructured
    This class defines the basic structure of a dual-input metric learning network.
    The two feature extraction modals needs implementation.
    The descriptors produced by the two feature extraction modals should have exact same dimensions.
    """

    def __init__(self, batch_size, desc_dims, out_dims, is_training=False, is_retrieving=False):
        lib.mlnet.MLNet.__init__(self, batch_size, desc_dims, out_dims, is_training, is_retrieving)

    def build(self, modal_1_device=None, modal_2_device=None, metrics_device=None):

        self.ph_dropout = tf.placeholder(tf.float32, [], 'dropout')
        self.ph_labels = tf.placeholder(tf.int32, [self.batch_size], 'labels')

        if self.is_retrieving:
            self.ph_desc_1 = tf.placeholder(tf.float32, [self.batch_size, self.desc_dims], 'desc_1')
            self.ph_desc_2 = tf.placeholder(tf.float32, [self.batch_size, self.desc_dims], 'desc_2')

        with tf.variable_scope('modal_1'), tf.device(modal_1_device):
            self.ph1, self.modal_1 = self.build_modal_1()
            _, F_m1 = self.modal_1.get_shape()
            self.descriptors_1 = self.modal_1

        with tf.variable_scope('modal_2'), tf.device(modal_2_device):
            self.ph2, self.modal_2 = self.build_modal_2()
            _, F_m2 = self.modal_2.get_shape()
            self.descriptors_2 = self.modal_2

        desc_1 = self.ph_desc_1 if self.is_retrieving else self.descriptors_1
        desc_2 = self.ph_desc_2 if self.is_retrieving else self.descriptors_2

        with tf.device(metrics_device):
            with tf.variable_scope('metrics'):
                x = tf.multiply(desc_1, desc_2)
                if self.is_training:
                    x = tf.nn.dropout(x, self.ph_dropout)
                with tf.variable_scope('fc'):
                    x, regularizers = self.fc(x, self.out_dims, activation_fn=None)
                    self.regularizers += regularizers
                self.logits = tf.squeeze(x)

        self.build_saver()

    def build_loss(self, lamda, mu, reg_weight):
        """Adds to the inference model the layers required to generate loss."""

        with tf.name_scope('loss'):
            with tf.name_scope('var_loss'):
                labels = tf.cast(self.ph_labels, tf.float32)
                shape = labels.get_shape()

                same_class = tf.boolean_mask(self.logits, tf.equal(labels, tf.ones(shape)))
                diff_class = tf.boolean_mask(self.logits, tf.not_equal(labels, tf.ones(shape)))
                same_mean, same_var = tf.nn.moments(same_class, [0])
                diff_mean, diff_var = tf.nn.moments(diff_class, [0])
                var_loss = same_var + diff_var

            with tf.name_scope('mean_loss'):
                mean_loss = lamda * tf.where(
                    tf.greater(mu - (same_mean - diff_mean), 0),
                    mu - (same_mean - diff_mean), 0)

            self.loss = (1) * var_loss + (1) * mean_loss
            regularize, regularization = len(self.regularizers) > 0, None
            if regularize:
                with tf.name_scope('regularization'):
                    regularization = reg_weight * tf.add_n(self.regularizers)
                self.loss += regularization

            # Summaries for TensorBoard.
            tf.summary.scalar('total', self.loss)
            with tf.name_scope('averages'):
                averages = tf.train.ExponentialMovingAverage(0.9)
                if regularize:
                    operations = [var_loss, mean_loss, regularization, self.loss]
                else:
                    operations = [var_loss, mean_loss, self.loss]
                op_averages = averages.apply(operations)
                tf.summary.scalar('var_loss', averages.average(var_loss))
                tf.summary.scalar('mean_loss', averages.average(mean_loss))
                if regularize:
                    tf.summary.scalar('regularization', averages.average(regularization))
                tf.summary.scalar('total', averages.average(self.loss))
                with tf.control_dependencies([op_averages]):
                    self.loss_average = tf.identity(averages.average(self.loss), name='control')