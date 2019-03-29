# -*- coding:utf-8 -*-

import tensorflow as tf
from lib.nn import NN
import numpy as np

def concat(x, R, A, axis):
    B, N, F_in = x.get_shape()
    x_exp = tf.expand_dims(x, 2)
    x_exp = tf.tile(x_exp, [1, 1, N, 1])
    fused = tf.concat([x_exp, R], axis=axis)
    B, N, N, F_f = fused.get_shape()
    A_exp = tf.expand_dims(A, 3)
    A_exp = tf.tile(A_exp, [1, 1, 1, F_f])
    A_exp = tf.cast(A_exp, tf.float32)
    fused = tf.multiply(fused, A_exp)
    return fused, []

def concat_i(i, x, R, A, axis):
    B, N, F_in = x.get_shape()
    R_i = R[:, :, i, :] # B * N * F_r
    A_i = A[:, :, i]
    fused = tf.concat([x, R_i], axis=axis)
    B, N, F_f = fused.get_shape()
    A_i_exp = tf.expand_dims(A_i, 2)
    A_i_exp = tf.tile(A_i_exp, [1, 1, F_f])
    A_i_exp = tf.cast(A_i_exp, tf.float32)
    fused = tf.multiply(fused, A_i_exp)
    return fused, []


class RGCN(NN):

    def __init__(self, max_nodes):
        NN.__init__(self)
        self.max_nodes = max_nodes

    def sdpatt(self, q, k, v, F_hid, n_att=1):

        B, M, F_q = q.get_shape()
        B, N, F_k = k.get_shape()
        B, N, F_v = v.get_shape()
        scalar = float(int(F_k)) ** 0.5

        with tf.name_scope('Q'):
            q_exp = tf.reshape(q, [B * M, F_q])
            q_proj, regs = self.fc(q_exp, F_hid, activation_fn=None, regularize=False)
            q = tf.reshape(q_proj, [B, M, F_hid])
            q_exp = tf.expand_dims(q, axis=2)
            q_exp = tf.expand_dims(q_exp, axis=3)
            q_exp = tf.tile(q_exp, [1, 1, N, n_att, 1])  # [B, M, N, n_att, F_q]

        with tf.name_scope('K'):
            v_exp = tf.reshape(v, [B * M, F_q])
            v_proj, regs = self.fc(v_exp, F_hid, activation_fn=None, regularize=False)
            v = tf.reshape(v_proj, [B, M, F_hid])
            k_exp = tf.expand_dims(k, axis=1)
            k_exp = tf.expand_dims(k_exp, axis=3)
            k_exp = tf.tile(k_exp, [1, M, 1, n_att, 1])  # [B, M, N, n_att, F_k]

        with tf.name_scope('V'):
            v_exp = tf.expand_dims(v, axis=1)
            v_exp = tf.expand_dims(v_exp, axis=3)
            v_exp = tf.tile(v_exp, [1, M, 1, n_att, 1])  # [B, M, N, n_att, F_v]

        with tf.name_scope('dot'):
            att = tf.multiply(q_exp, k_exp)
            att = tf.reduce_sum(att, axis=-1)  # [B, M, N, n_att]
            att = att / scalar
            tf.summary.histogram('att_weights', att)

        with tf.name_scope('softmax'):
            att = tf.nn.softmax(att, 2)  # [B, M, N, n_att]

        with tf.name_scope('attend'):
            att_exp = tf.expand_dims(att, axis=4)
            att_exp = tf.tile(att_exp, [1, 1, 1, 1, F_v])
            v_att = tf.multiply(v_exp, att_exp)  # [B, M, N, n_att, F_v]
            v_att = tf.reduce_sum(v_att, axis=2)  # [B, M, n_att, F_v]
            v_att = tf.reshape(v_att, [B, M, n_att * int(F_v)])

        return v_att

    def rgconv_att(self, x, R, F_out, n_att=1, fuse_fn=concat, activation_fn=tf.nn.relu, batch_norm=True, regularize=False):

        if F_out % n_att != 0: raise Exception("F_out must be dividable by n_att")
        F_out_per = int(F_out / n_att)

        B, N, F_in = x.get_shape()
        if N != self.max_nodes:
            raise Exception("input must match the dimensions of max_nodes")

        regularizers = []

        A = np.ones([B, N, N], dtype='int32')
        A[:, :, 0] = 0
        A[:, 0, :] = 0
        A = tf.convert_to_tensor(A)

        with tf.variable_scope('fuse'):
            fused, regs = fuse_fn(x, R, A, 3)
            regularizers += regs
            B, N, N, F_in = fused.get_shape()
            f_flat = tf.reshape(fused, [B * N * N, F_in])

        with tf.variable_scope('convolve'):
            W, regs = self.weight_variable([F_in, F_out_per], regularize=regularize)
            regularizers += regs
            b, regs = self.bias_variable([1, F_out_per], regularize=regularize)
            regularizers += regs
            h_flat = tf.matmul(f_flat, W) + b

        with tf.variable_scope('attention'):
            B, N, N, F_r = R.get_shape()
            R_flat = tf.reshape(R, [B*N*N, F_r])
            corr, regs = self.fc(R_flat, n_att, activation_fn=None, regularize=False)
            regularizers += regs
            corr = tf.reshape(corr, [B, N, N, n_att])
            att = tf.nn.softmax(corr, 1)
            att = tf.expand_dims(att, axis=4)
            att = tf.tile(att, [1, 1, 1, 1, F_out_per])

        with tf.variable_scope('attend'):
            h = tf.reshape(h_flat, [B, N, N, 1, F_out_per])
            h = tf.tile(h, [1, 1, 1, n_att, 1])
            h = tf.multiply(h, att)
            h = tf.reshape(h, [B, N, N, F_out])
            h = tf.reduce_sum(h, axis=1)

        h = tf.layers.batch_normalization(h) if batch_norm else h
        h = activation_fn(h) if activation_fn is not None else h

        return h, regularizers

    def rgconv_i_att(self, i, x, R, F_out, n_att=1, fuse_fn=concat_i, activation_fn=tf.nn.relu, batch_norm=True, dropout=None, regularize=False):

        if F_out % n_att != 0: raise Exception("F_out must be dividable by n_att")
        F_out_per = int(F_out / n_att)

        B, N, F_in = x.get_shape()
        if N != self.max_nodes:
            raise Exception("input must match the dimensions of max_nodes")

        regularizers = []

        A = np.zeros([B, N, N], dtype='int32')
        A[:, :, i] = 1
        A[:, i, i] = 0
        A = tf.convert_to_tensor(A)

        with tf.variable_scope('fuse'):
            fused, regs = fuse_fn(i, x, R, A, 2)
            regularizers += regs
            B, N, F_in = fused.get_shape()
            f_flat = tf.reshape(fused, [B * N, F_in])

        f_flat = tf.nn.dropout(f_flat, dropout) if dropout else f_flat

        with tf.variable_scope('convolve'):
            W, regs = self.weight_variable([F_in, F_out_per], regularize=regularize)
            regularizers += regs
            b, regs = self.bias_variable([1, F_out_per], regularize=regularize)
            regularizers += regs
            h_flat = tf.matmul(f_flat, W) + b

        with tf.variable_scope('attention'):
            saliency, regs = self.fc(f_flat, n_att, activation_fn=None, regularize=False)
            saliency = saliency + 0.5
            regularizers += regs
            saliency = tf.reshape(saliency, [B, N, n_att])
            att = tf.nn.softmax(saliency, 1)
            att = tf.expand_dims(att, axis=3)
            att = tf.tile(att, [1, 1, 1, F_out_per])

        with tf.variable_scope('attend'):
            h = tf.reshape(h_flat, [B, N, 1, F_out_per])
            h = tf.tile(h, [1, 1, n_att, 1])
            h = tf.multiply(h, att)
            h = tf.reshape(h, [B, N, F_out])
            h = tf.reduce_sum(h, axis=1)

        h = tf.layers.batch_normalization(h) if batch_norm else h
        h = activation_fn(h) if activation_fn is not None else h

        return h, regularizers

    def gconv_i_att(self, i, x, F_out, n_att=1, fuse_fn=concat_i, activation_fn=tf.nn.relu, batch_norm=True, dropout=None, regularize=False):

        if F_out % n_att != 0: raise Exception("F_out must be dividable by n_att")
        F_out_per = int(F_out / n_att)

        B, N, F_in = x.get_shape()
        if N != self.max_nodes:
            raise Exception("input must match the dimensions of max_nodes")

        regularizers = []

        A = np.zeros([B, N, N], dtype='int32')
        A[:, :, i] = 1
        A[:, i, i] = 0
        A = tf.convert_to_tensor(A)

        with tf.variable_scope('fuse'):
            R = tf.zeros([B, N, N, 0])
            fused, regs = fuse_fn(i, x, R, A, 2)
            regularizers += regs
            B, N, F_in = fused.get_shape()
            f_flat = tf.reshape(fused, [B * N, F_in])

        f_flat = tf.nn.dropout(f_flat, dropout) if dropout else f_flat

        with tf.variable_scope('convolve'):
            W, regs = self.weight_variable([F_in, F_out_per], regularize=regularize)
            regularizers += regs
            b, regs = self.bias_variable([1, F_out_per], regularize=regularize)
            regularizers += regs
            h_flat = tf.matmul(f_flat, W) + b

        with tf.variable_scope('attention'):
            saliency, regs = self.fc(f_flat, n_att, activation_fn=None, regularize=False)
            saliency = saliency + 0.5
            regularizers += regs
            saliency = tf.reshape(saliency, [B, N, n_att])
            att = tf.nn.softmax(saliency, 1)
            att = tf.expand_dims(att, axis=3)
            att = tf.tile(att, [1, 1, 1, F_out_per])

        with tf.variable_scope('attend'):
            h = tf.reshape(h_flat, [B, N, 1, F_out_per])
            h = tf.tile(h, [1, 1, n_att, 1])
            h = tf.multiply(h, att)
            h = tf.reshape(h, [B, N, F_out])
            h = tf.reduce_sum(h, axis=1)

        h = tf.layers.batch_normalization(h) if batch_norm else h
        h = activation_fn(h) if activation_fn is not None else h

        return h, regularizers

    def gconv_att(self, x, F_out, n_att=1, fuse_fn=concat, activation_fn=tf.nn.relu, batch_norm=True, regularize=False):

        if F_out % n_att != 0: raise Exception("F_out must be dividable by n_att")
        F_out_per = int(F_out / n_att)

        B, N, F_in = x.get_shape()
        if N != self.max_nodes:
            raise Exception("input must match the dimensions of max_nodes")

        regularizers = []

        A = np.ones([B, N, N], dtype='int32')
        A[:, :, 0] = 0
        A[:, 0, :] = 0
        A = tf.convert_to_tensor(A)

        with tf.variable_scope('fuse'):
            R = tf.zeros([B, N, N, 0])
            fused, regs = fuse_fn(x, R, A, 3)
            regularizers += regs
            B, N, N, F_in = fused.get_shape()
            f_flat = tf.reshape(fused, [B * N * N, F_in])

        with tf.variable_scope('convolve'):
            W, regs = self.weight_variable([F_in, F_out_per], regularize=regularize)
            regularizers += regs
            b, regs = self.bias_variable([1, F_out_per], regularize=regularize)
            regularizers += regs
            h_flat = tf.matmul(f_flat, W) + b

        with tf.variable_scope('attention'):
            corr, regs = self.fc(f_flat, n_att, activation_fn=None, regularize=False)
            regularizers += regs
            corr = tf.reshape(corr, [B, N, N, n_att])
            att = tf.nn.softmax(corr, 1)
            att = tf.expand_dims(att, axis=4)
            att = tf.tile(att, [1, 1, 1, 1, F_out_per])

        with tf.variable_scope('attend'):
            h = tf.reshape(h_flat, [B, N, N, 1, F_out_per])
            h = tf.tile(h, [1, 1, 1, n_att, 1])
            h = tf.multiply(h, att)
            h = tf.reshape(h, [B, N, N, F_out])
            h = tf.reduce_sum(h, axis=1)

        h = tf.layers.batch_normalization(h) if batch_norm else h
        h = activation_fn(h) if activation_fn is not None else h

        return h, regularizers

    def gconv(self, x, A, F_out, fuse_fn=concat, activation_fn=tf.nn.relu, batch_norm=True,
                  regularize=False):


        B, N, F_in = x.get_shape()
        if N != self.max_nodes:
            raise Exception("input must match the dimensions of max_nodes")

        regularizers = []

        with tf.variable_scope('fuse'):
            R = tf.zeros([B, N, N, 0])
            fused, regs = fuse_fn(x, R, A, 3)
            regularizers += regs
            B, N, N, F_in = fused.get_shape()
            f_flat = tf.reshape(fused, [B * N * N, F_in])

        with tf.variable_scope('convolve'):
            W, regs = self.weight_variable([F_in, F_out], regularize=regularize)
            regularizers += regs
            b, regs = self.bias_variable([1, F_out], regularize=regularize)
            regularizers += regs
            h_flat = tf.matmul(f_flat, W) + b
            h = tf.reshape(h_flat, [B, N, N, F_out])
            h = tf.reduce_sum(h, axis=1)

        h = tf.layers.batch_normalization(h) if batch_norm else h
        h = activation_fn(h) if activation_fn is not None else h

        return h, regularizers

    def gconv_i(self, i, x, F_out, n_att=1, fuse_fn=concat_i, activation_fn=tf.nn.relu, batch_norm=True, dropout=None, regularize=False):

        if F_out % n_att != 0: raise Exception("F_out must be dividable by n_att")
        F_out_per = int(F_out / n_att)

        B, N, F_in = x.get_shape()
        if N != self.max_nodes:
            raise Exception("input must match the dimensions of max_nodes")

        regularizers = []

        A = np.zeros([B, N, N], dtype='int32')
        A[:, :, i] = 1
        A[:, i, i] = 0
        A = tf.convert_to_tensor(A)

        with tf.variable_scope('fuse'):
            R = tf.zeros([B, N, N, 0])
            fused, regs = fuse_fn(i, x, R, A, 2)
            regularizers += regs
            B, N, F_in = fused.get_shape()
            f_flat = tf.reshape(fused, [B * N, F_in])

        f_flat = tf.nn.dropout(f_flat, dropout) if dropout else f_flat

        with tf.variable_scope('convolve'):
            W, regs = self.weight_variable([F_in, F_out_per], regularize=regularize)
            regularizers += regs
            b, regs = self.bias_variable([1, F_out_per], regularize=regularize)
            regularizers += regs
            h_flat = tf.matmul(f_flat, W) + b
            h = tf.reshape(h_flat, [B, N, F_out])
            h = tf.reduce_sum(h, axis=1)

        h = tf.layers.batch_normalization(h) if batch_norm else h
        h = activation_fn(h) if activation_fn is not None else h

        return h, regularizers

    def gconv_l_att(self, x, A, F_out, n_att=1, fuse_fn=concat, activation_fn=tf.nn.relu, batch_norm=True, regularize=False):

        if F_out % n_att != 0: raise Exception("F_out must be dividable by n_att")
        F_out_per = int(F_out / n_att)

        B, N, F_in = x.get_shape()
        if N != self.max_nodes:
            raise Exception("input must match the dimensions of max_nodes")

        regularizers = []

        with tf.variable_scope('fuse'):
            R = tf.zeros([B, N, N, 0])
            fused, regs = fuse_fn(x, R, A, 3)
            regularizers += regs
            B, N, N, F_in = fused.get_shape()
            f_flat = tf.reshape(fused, [B * N * N, F_in])

        with tf.variable_scope('convolve'):
            W, regs = self.weight_variable([F_in, F_out_per], regularize=regularize)
            regularizers += regs
            b, regs = self.bias_variable([1, F_out_per], regularize=regularize)
            regularizers += regs
            h_flat = tf.matmul(f_flat, W) + b

        with tf.variable_scope('attention'):
            corr, regs = self.fc(f_flat, n_att, activation_fn=None, regularize=False)
            regularizers += regs
            corr = tf.reshape(corr, [B, N, N, n_att])
            att = tf.nn.softmax(corr, 1)
            att = tf.expand_dims(att, axis=4)
            att = tf.tile(att, [1, 1, 1, 1, F_out_per])

        with tf.variable_scope('attend'):
            h = tf.reshape(h_flat, [B, N, N, 1, F_out_per])
            h = tf.tile(h, [1, 1, 1, n_att, 1])
            h = tf.multiply(h, att)
            h = tf.reshape(h, [B, N, N, F_out])
            h = tf.reduce_sum(h, axis=1)

        h = tf.layers.batch_normalization(h) if batch_norm else h
        h = activation_fn(h) if activation_fn is not None else h

        return h, regularizers