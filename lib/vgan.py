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


class VGAN(NN):

    def __init__(self, max_nodes):
        NN.__init__(self)
        self.max_nodes = max_nodes

    def rgconv_att(self, x, F_out, n_att=1, activation_fn=tf.nn.relu, batch_norm=True, regularize=False):

        if F_out % n_att != 0: raise Exception("F_out must be dividable by n_att")
        F_out_per = int(F_out / n_att)

        B, N, F_in = x.get_shape()
        if N != self.max_nodes:
            raise Exception("input must match the dimensions of max_nodes")

        regularizers = []

        x_flat = tf.reshape(x, [B*N, F_in])

        with tf.variable_scope('convolve'):
            W, regs = self.weight_variable([F_in, F_out], regularize=regularize)
            regularizers += regs
            b, regs = self.bias_variable([1, F_out], regularize=regularize)
            regularizers += regs
            h_flat = tf.matmul(x_flat, W) + b

        with tf.variable_scope('attention'):
            corr, regs = self.fc(x_flat, n_att, activation_fn=None, regularize=False)
            corr = corr + 0.5
            regularizers += regs
            corr = tf.reshape(corr, [B, N, n_att])
            att = tf.nn.softmax(corr, 1)
            att = tf.expand_dims(att, axis=2)
            att = tf.expand_dims(att, axis=4)
            att = tf.tile(att, [1, 1, N, 1, F_out_per])

        with tf.variable_scope('attend'):
            h_flat = tf.reshape(h_flat, [B, N, n_att, F_out_per])
            h_flat = tf.expand_dims(h_flat, 1)
            h_flat = tf.tile(h_flat, [1, N, 1, 1, 1])
            h = tf.multiply(h_flat, att)
            h = tf.reduce_sum(h, axis=1)
            h = tf.reshape(h, [B, N, F_out])

        h = tf.layers.batch_normalization(h) if batch_norm else h
        h = activation_fn(h) if activation_fn is not None else h

        return h, regularizers

    def gconv_i_att(self, x, F_out, n_att=1, fuse_fn=concat_i, activation_fn=tf.nn.relu, batch_norm=True, dropout=None, regularize=False):

        if F_out % n_att != 0: raise Exception("F_out must be dividable by n_att")
        F_out_per = int(F_out / n_att)

        B, N, F_in = x.get_shape()
        if N != self.max_nodes:
            raise Exception("input must match the dimensions of max_nodes")

        regularizers = []

        x_flat = tf.reshape(x, [B*N, F_in])
        x_flat = tf.nn.dropout(x_flat, dropout) if dropout else x_flat

        with tf.variable_scope('convolve'):
            W, regs = self.weight_variable([F_in, F_out], regularize=regularize)
            regularizers += regs
            b, regs = self.bias_variable([1, F_out], regularize=regularize)
            regularizers += regs
            h_flat = tf.matmul(x_flat, W) + b

        with tf.variable_scope('attention'):
            saliency, regs = self.fc(x_flat, n_att, activation_fn=tf.nn.relu, regularize=False)
            saliency = saliency + 0.5
            regularizers += regs
            saliency = tf.reshape(saliency, [B, N, n_att])
            att = tf.nn.softmax(saliency, 1)
            att = tf.expand_dims(att, axis=3)
            att = tf.tile(att, [1, 1, 1, F_out_per])

        with tf.variable_scope('attend'):
            h = tf.reshape(h_flat, [B, N, n_att, F_out_per])
            h = tf.multiply(h, att)
            h = tf.reshape(h, [B, N, F_out])
            h = tf.reduce_sum(h, axis=1)

        h = tf.layers.batch_normalization(h) if batch_norm else h
        h = activation_fn(h) if activation_fn is not None else h

        return h, regularizers