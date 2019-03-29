from lib.mlnet_rs import MLNet
from modules.vgg19 import Vgg19
from modules.vgg16_trainable import Vgg16
from lib.gcn import gcn_specs, GCN
from scipy import sparse
import modules.graph as graph
import tensorflow as tf
import numpy as np

class Model(MLNet):
    """
    A baseline model for experimentation on eng-wiki

    modal 1:
        input: extracted text tf-idf features
        method: tf_idf
        output: tf_idf features

    modal 2:
        input: VGG19 extracted pool5 features
        method: VGG19 + gcn
        output: gcn features

    conventions:
        input of a graph convolution layer is usually a NxMxF tensor
        N: number of samples (texts, images, etc.)
        M: number of graph nodes in each sample
        F: number of feature dimensions on each node
    """

    def __init__(self, max_rgcn_nodes, batch_size, desc_dims, out_dims, is_training=False, is_retrieving=False):
        MLNet.__init__(self, batch_size, desc_dims, out_dims, is_training, is_retrieving)
        self.max_rgcn_nodes = max_rgcn_nodes

    def build_modal_1(self):
        M_text = 4412

        ph_text = tf.placeholder(tf.float32, [self.batch_size, M_text], 'input_text')

        with tf.variable_scope('fc_1'):
            fc_1, regularizers = self.fc(ph_text, self.desc_dims, activation_fn=tf.nn.relu, regularize=False)
            self.regularizers += regularizers

        with tf.variable_scope('fc_2'):
            fc_2, regularizers = self.fc(fc_1, self.desc_dims, activation_fn=tf.nn.relu, regularize=False)
            self.regularizers += regularizers

        return [ph_text], fc_2

    def build_modal_2(self):
        B = self.batch_size
        N = self.max_rgcn_nodes
        F_in = 2048
        F_r = 128
        dropout_ratio = 0.5
        ph_x = tf.placeholder(tf.float32, [B, N, F_in], name='image_feature')
        x_in = tf.layers.batch_normalization(ph_x, center=False)

        ph_R = tf.placeholder(tf.float32, [B, N, N, F_r], name='relation_feature')
        R_in = tf.layers.batch_normalization(ph_R, center=False)

        ph_A = tf.placeholder(tf.int32, [B, N, N], name='adj_mat')

        subsapce_dim = 384
        relation_glimpse = 1  # output dimension of RNs

        # Image matrix translation X
        # ph_image = tf.placeholder(tf.float32, [self.batch_size, N, input_dim], 'input_image')
        print('ph_x',ph_x.get_shape())
        ph_reshape = tf.reshape(ph_x, [int(B * N), F_in])
        ph_subdim, sm_reg = self.fc(ph_reshape, subsapce_dim, activation_fn=None)
        self.regularizers += sm_reg
        ph_subdim = tf.reshape(ph_subdim, [int(B), int(N), int(subsapce_dim)])

        ph_exp1 = tf.expand_dims(ph_subdim, 1)
        ph_exp1 = tf.tile(ph_exp1, [1, N, 1, 1])
        ph_exp2 = tf.expand_dims(ph_subdim, 2)
        ph_exp2 = tf.tile(ph_exp2, [1, 1, N, 1])
        ph_input = ph_exp1 * ph_exp2  # [bs,N,N, ph_subdim]

        X0 = tf.nn.dropout(tf.nn.relu(tf.layers.conv2d(inputs=ph_input,filters=int(subsapce_dim/2),kernel_size=1)),dropout_ratio)
        X0 = tf.nn.dropout(tf.nn.relu(tf.layers.conv2d(inputs=X0,filters=int(subsapce_dim/4),kernel_size=1)),dropout_ratio)
        X0 = tf.nn.dropout(tf.nn.relu(tf.layers.conv2d(inputs=X0,filters=relation_glimpse,kernel_size=1)), dropout_ratio)
        rel_map0 = X0 + tf.transpose(X0,[0,2,1,3])
        # print('rel_map0_bt',rel_map0.get_shape())
        rel_map0 = tf.transpose(rel_map0,[0,3,2,1])
        # print('rel_map0_at',rel_map0.get_shape())

        rel_map0 = tf.reshape(rel_map0,[self.batch_size,relation_glimpse,-1])
        # print('rel_map0_shape1',rel_map0.get_shape())
        rel_map0 = tf.nn.softmax(rel_map0,axis=2)
        rel_map0 = tf.reshape(rel_map0,[self.batch_size,relation_glimpse,N,-1])
        print('rel_map0',rel_map0.get_shape())

        # X1 = tf.nn.dropout(tf.nn.relu(tf.layers.conv2d(inputs=ph_input,filters=int(subsapce_dim/2),kernel_size=1,dilation_rate=(1,1),padding='valid')),dropout_ratio)
        # X1 = tf.nn.dropout(tf.nn.relu(tf.layers.conv2d(inputs=X1,filters=int(subsapce_dim/4),kernel_size=1,dilation_rate=(1,2),padding='valid')),dropout_ratio)
        # X1 = tf.nn.dropout(tf.nn.relu(tf.layers.conv2d(inputs=X1,filters=relation_glimpse,kernel_size=1,dilation_rate=(1,4),padding='valid')),dropout_ratio)
        
        X1 = tf.nn.dropout(tf.nn.relu(tf.layers.conv2d(inputs=ph_input,filters=int(subsapce_dim/2),kernel_size=3,dilation_rate=(1,1),padding='same')),dropout_ratio)
        X1 = tf.nn.dropout(tf.nn.relu(tf.layers.conv2d(inputs=X1,filters=int(subsapce_dim/4),kernel_size=3,dilation_rate=(1,2),padding='same')),dropout_ratio)
        X1 = tf.nn.dropout(tf.nn.relu(tf.layers.conv2d(inputs=X1,filters=relation_glimpse,kernel_size=3,dilation_rate=(1,4),padding='same')),dropout_ratio)
        
        rel_map1 = X1 + tf.transpose(X1,[0,2,1,3])
        rel_map1 = tf.transpose(rel_map1,[0,3,2,1])
        rel_map1 = tf.reshape(rel_map1,[self.batch_size,relation_glimpse,-1])
        rel_map1 = tf.nn.softmax(rel_map1,2)
        rel_map1 = tf.reshape(rel_map1,[self.batch_size,relation_glimpse,N,-1])
        
        print('rel_map1',rel_map1.get_shape())
        print('ph_x',ph_x.get_shape())

        rel_x = tf.zeros_like(ph_x)
        for g in range(relation_glimpse):
            rel_x = rel_x + tf.matmul(rel_map1[:,g,:,:], ph_x) + tf.matmul(rel_map0[:,g,:,:], ph_x)
        rel_x = rel_x/(2 * relation_glimpse)

        rn_out = tf.reshape(rel_x,[self.batch_size,-1])
        
        fc_out = tf.squeeze(tf.reduce_sum(ph_x,1))

        with tf.variable_scope('rn_fc_2'):
            rn_out_2, regularizers = self.fc(rn_out, self.desc_dims, activation_fn=tf.nn.relu, regularize=True)
            self.regularizers += regularizers

        # +================================================================================================
        # with tf.variable_scope('obj_proj'):
        #     x_reshape = tf.reshape(x_in, [B*N, F_in])
        #     x_proj, regs = self.fc(x_reshape, 512, tf.nn.relu, regularize=False)
        #     x_proj = tf.reshape(x_proj, [B, N, 512])
        #     self.regularizers += regs

        # rgcn = RGCN(N)

        # with tf.variable_scope('rgcn_gconv1'):
        #     rgconv1, regs = rgcn.rgconv_att(x_proj, R_in, ph_A, 256, regularize=False)
        # self.regularizers += regs

        # with tf.variable_scope('rgcn_gconv2'):
        #     rgconv2, regs = rgcn.rgconv_att(rgconv1, R_in, ph_A, 256, regularize=False)
        # self.regularizers += regs

        # with tf.variable_scope('rgcn_rgconv_assem'):
        #     keep_prob = 0.8 if self.is_training else 1
        #     rgcn_out, regs = rgcn.rgconv_i_att(0, rgconv2, R_in, 512, dropout=keep_prob, regularize=False)
        # self.regularizers += regs

        return [ph_x, ph_R, ph_A], rn_out_2
