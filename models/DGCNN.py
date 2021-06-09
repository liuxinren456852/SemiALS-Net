import tensorflow as tf
import math
import time
import numpy as np
import os
import sys

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(BASE_DIR))
# sys.path.append(os.path.join(BASE_DIR,'../Networks/dgcnn/utils'))
# sys.path.append(os.path.join(BASE_DIR,'../Networks/dgcnn/models'))

import dgcnn_util as tf_util

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 5))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    smpws_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl, smpws_pl


def get_model(point_cloud, is_training, num_class, weight_decay=0., bn_decay=None):
    """ ConvNet baseline, input is BxNx9 gray image, output B x num_class """
    
    end_points = {}
    l0_xyz = tf.slice(point_cloud, [0, 0, 0], [-1, -1, 3])
    l0_points = tf.slice(point_cloud, [0, 0, 3], [-1, -1, 2])
    end_points['l0_xyz'] = l0_xyz
    
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    input_image = tf.expand_dims(point_cloud, -1)

    k = 20

    adj = tf_util.pairwise_distance(point_cloud[:, :, 3:])
    nn_idx = tf_util.knn(adj, k=k)  # (batch, num_points, k)
    edge_feature = tf_util.get_edge_feature(input_image, nn_idx=nn_idx, k=k)

    out1 = tf_util.conv2d(edge_feature, 64, [1, 1],
                          padding='VALID', stride=[1, 1],
                          bn=True, is_training=is_training, weight_decay=weight_decay,
                          scope='adj_conv1', bn_decay=bn_decay, is_dist=True)

    out2 = tf_util.conv2d(out1, 64, [1, 1],
                          padding='VALID', stride=[1, 1],
                          bn=True, is_training=is_training, weight_decay=weight_decay,
                          scope='adj_conv2', bn_decay=bn_decay, is_dist=True)

    net_1 = tf.reduce_max(out2, axis=-2, keep_dims=True)

    adj = tf_util.pairwise_distance(net_1)
    nn_idx = tf_util.knn(adj, k=k)
    edge_feature = tf_util.get_edge_feature(net_1, nn_idx=nn_idx, k=k)

    out3 = tf_util.conv2d(edge_feature, 64, [1, 1],
                          padding='VALID', stride=[1, 1],
                          bn=True, is_training=is_training, weight_decay=weight_decay,
                          scope='adj_conv3', bn_decay=bn_decay, is_dist=True)

    out4 = tf_util.conv2d(out3, 64, [1, 1],
                          padding='VALID', stride=[1, 1],
                          bn=True, is_training=is_training, weight_decay=weight_decay,
                          scope='adj_conv4', bn_decay=bn_decay, is_dist=True)

    net_2 = tf.reduce_max(out4, axis=-2, keep_dims=True)

    adj = tf_util.pairwise_distance(net_2)
    nn_idx = tf_util.knn(adj, k=k)
    edge_feature = tf_util.get_edge_feature(net_2, nn_idx=nn_idx, k=k)

    out5 = tf_util.conv2d(edge_feature, 64, [1, 1],
                          padding='VALID', stride=[1, 1],
                          bn=True, is_training=is_training, weight_decay=weight_decay,
                          scope='adj_conv5', bn_decay=bn_decay, is_dist=True)

    # out6 = tf_util.conv2d(out5, 64, [1,1],
    #                      padding='VALID', stride=[1,1],
    #                      bn=True, is_training=is_training, weight_decay=weight_decay,
    #                      scope='adj_conv6', bn_decay=bn_decay, is_dist=True)

    net_3 = tf.reduce_max(out5, axis=-2, keep_dims=True)

    out7 = tf_util.conv2d(tf.concat([net_1, net_2, net_3], axis=-1), 1024, [1, 1],
                          padding='VALID', stride=[1, 1],
                          bn=True, is_training=is_training,
                          scope='adj_conv7', bn_decay=bn_decay, is_dist=True)

    out_max = tf_util.max_pool2d(out7, [num_point, 1], padding='VALID', scope='maxpool')

    expand = tf.tile(out_max, [1, num_point, 1, 1])

    concat = tf.concat(axis=3, values=[expand,
                                       net_1,
                                       net_2,
                                       net_3])

    # CONV
    net = tf_util.conv2d(concat, 512, [1, 1], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='seg/conv1', is_dist=True)
    net = tf_util.conv2d(net, 256, [1, 1], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='seg/conv2', is_dist=True)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp1')
    net = tf_util.conv2d(net, num_class, [1, 1], padding='VALID', stride=[1, 1],
                         activation_fn=None, scope='seg/conv3', is_dist=True)
    net = tf.squeeze(net, [2])

    return net, end_points

def get_model_unnormXYZ(point_cloud, is_training, weight_decay=0., bn_decay=None):
    """ ConvNet baseline, input is BxNx9 gray image """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    input_image = tf.expand_dims(point_cloud, -1)

    k = 20

    adj = tf_util.pairwise_distance(point_cloud[:, :, 0:3])
    nn_idx = tf_util.knn(adj, k=k)  # (batch, num_points, k)
    edge_feature = tf_util.get_edge_feature(input_image, nn_idx=nn_idx, k=k)

    out1 = tf_util.conv2d(edge_feature, 64, [1, 1],
                          padding='VALID', stride=[1, 1],
                          bn=True, is_training=is_training, weight_decay=weight_decay,
                          scope='adj_conv1', bn_decay=bn_decay, is_dist=True)

    out2 = tf_util.conv2d(out1, 64, [1, 1],
                          padding='VALID', stride=[1, 1],
                          bn=True, is_training=is_training, weight_decay=weight_decay,
                          scope='adj_conv2', bn_decay=bn_decay, is_dist=True)

    net_1 = tf.reduce_max(out2, axis=-2, keep_dims=True)

    adj = tf_util.pairwise_distance(net_1)
    nn_idx = tf_util.knn(adj, k=k)
    edge_feature = tf_util.get_edge_feature(net_1, nn_idx=nn_idx, k=k)

    out3 = tf_util.conv2d(edge_feature, 64, [1, 1],
                          padding='VALID', stride=[1, 1],
                          bn=True, is_training=is_training, weight_decay=weight_decay,
                          scope='adj_conv3', bn_decay=bn_decay, is_dist=True)

    out4 = tf_util.conv2d(out3, 64, [1, 1],
                          padding='VALID', stride=[1, 1],
                          bn=True, is_training=is_training, weight_decay=weight_decay,
                          scope='adj_conv4', bn_decay=bn_decay, is_dist=True)

    net_2 = tf.reduce_max(out4, axis=-2, keep_dims=True)

    adj = tf_util.pairwise_distance(net_2)
    nn_idx = tf_util.knn(adj, k=k)
    edge_feature = tf_util.get_edge_feature(net_2, nn_idx=nn_idx, k=k)

    out5 = tf_util.conv2d(edge_feature, 64, [1, 1],
                          padding='VALID', stride=[1, 1],
                          bn=True, is_training=is_training, weight_decay=weight_decay,
                          scope='adj_conv5', bn_decay=bn_decay, is_dist=True)

    # out6 = tf_util.conv2d(out5, 64, [1,1],
    #                      padding='VALID', stride=[1,1],
    #                      bn=True, is_training=is_training, weight_decay=weight_decay,
    #                      scope='adj_conv6', bn_decay=bn_decay, is_dist=True)

    net_3 = tf.reduce_max(out5, axis=-2, keep_dims=True)

    out7 = tf_util.conv2d(tf.concat([net_1, net_2, net_3], axis=-1), 1024, [1, 1],
                          padding='VALID', stride=[1, 1],
                          bn=True, is_training=is_training,
                          scope='adj_conv7', bn_decay=bn_decay, is_dist=True)

    out_max = tf_util.max_pool2d(out7, [num_point, 1], padding='VALID', scope='maxpool')

    expand = tf.tile(out_max, [1, num_point, 1, 1])

    concat = tf.concat(axis=3, values=[expand,
                                       net_1,
                                       net_2,
                                       net_3])

    # CONV
    net = tf_util.conv2d(concat, 512, [1, 1], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='seg/conv1', is_dist=True)
    net = tf_util.conv2d(net, 256, [1, 1], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='seg/conv2', is_dist=True)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp1')
    net = tf_util.conv2d(net, 13, [1, 1], padding='VALID', stride=[1, 1],
                         activation_fn=None, scope='seg/conv3', is_dist=True)
    net = tf.squeeze(net, [2])

    return net


def get_loss(pred, label, smpw):
    """
    :param pred: BxNxC
    :param label: BxN
    :param smpw: BxN
    :return:
    """
    # Deep supervision
    classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=label, logits=pred, weights=smpw)
    tf.summary.scalar('classify_loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    
    return classify_loss