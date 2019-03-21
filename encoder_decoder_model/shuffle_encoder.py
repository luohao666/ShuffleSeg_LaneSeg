#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-1-29 下午2:04
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : dilation_encoder.py
# @IDE: PyCharm Community Edition
"""
实现一个基于VGG16的特征编码类
"""
from collections import OrderedDict

import tensorflow as tf

from encoder_decoder_model import cnn_basenet


class ShuffleEncoder(cnn_basenet.CNNBaseModel):
    """
    实现了一个基于ShuffleNet的特征编码类
    """
    def __init__(self, phase):
        """

        :param phase:
        """
        super(ShuffleEncoder, self).__init__()
        self._train_phase = tf.constant('train', dtype=tf.string)
        self._test_phase = tf.constant('test', dtype=tf.string)
        self._phase = phase
        self._is_training = self._init_phase()

        # A number stands for the num_groups
        # Output channels for conv1 layer
        self.output_channels = {'1': [144, 288, 576], '2': [200, 400, 800], '3': [240, 480, 960], '4': [272, 544, 1088],
                                '8': [384, 768, 1536], 'conv1': 24}
        self.num_classes=2


    def _init_phase(self):
        """
        :return:
        """
        """ 比较是否相等"""
        return tf.equal(self._phase, self._train_phase)


    def _conv_stage(self, input_tensor, k_size, out_dims, name,
                    stride=1, pad='SAME'):
        """
        将卷积和激活封装在一起
        :param input_tensor:
        :param k_size:
        :param out_dims:
        :param name:
        :param stride:
        :param pad:
        :return:
        """
        with tf.variable_scope(name):
            conv = self.conv2d(inputdata=input_tensor, out_channel=out_dims,
                               kernel_size=k_size, stride=stride,
                               use_bias=False, padding=pad, name='conv')

            bn = self.layerbn(inputdata=conv, is_training=self._is_training, name='bn')

            relu = self.relu(inputdata=bn, name='relu')

            return relu

    #实现群卷积GC
    def _grouped_conv2d(self, x, name, k_size=3, num_filters=24, num_groups=4, stride=1, pad='SAME',activation=None, batchnorm_enabled=False):
       
        with tf.variable_scope(name) as scope:
            sz = x.get_shape()[3].value // num_groups
            conv_side_layers = [
                self.conv2d(inputdata=x[:, :, :, i * sz:i * sz + sz], out_channel=num_filters // num_groups,
                               kernel_size=k_size, stride=stride,
                               use_bias=False, padding=pad, name=name + "_" + str(i)) for i in range(num_groups)]
            conv_g = tf.concat(conv_side_layers, axis=-1)
#            print(conv_side_layers[0].shape)
#            print(conv_side_layers[1].shape)
#            print(conv_side_layers[2].shape)
            if batchnorm_enabled:
                conv_o_bn = tf.layers.batch_normalization(conv_g, training=self._is_training, epsilon=1e-5)
                if not activation:
                    conv_a = conv_o_bn
                else:
                    conv_a = activation(conv_o_bn)
            else:
                if not activation:
                    conv_a = conv_g
                else:
                    conv_a = activation(conv_g)
            return conv_a

    #实现深度卷积DWC
    def _depthwise_conv2d(self, x, name, k_size=3, stride=1, pad='SAME',activation=None, batchnorm_enabled=False):
        with tf.variable_scope(name) as scope:
            conv_o_b = self.depthwise_conv2d_p(name=scope, inputdata=x, out_channel=x.shape[-1],kernel_size=k_size, padding=pad, stride=stride)

            if batchnorm_enabled:
                conv_o_bn = tf.layers.batch_normalization(conv_o_b, training=self._is_training, epsilon=1e-5)
                if not activation:
                    conv_a = conv_o_bn
                else:
                    conv_a = activation(conv_o_bn)
            else:
                if not activation:
                    conv_a = conv_o_b
                else:
                    conv_a = activation(conv_o_b)
            return conv_a

    #实现shuffle channel
    def _channel_shuffle(self,x,num_groups,name=None):
        with tf.variable_scope(name) as scope:
            n, h, w, c = x.shape.as_list()
            x_reshaped = tf.reshape(x, [-1, h, w, num_groups, c // num_groups])
            x_transposed = tf.transpose(x_reshaped, [0, 1, 2, 4, 3])
            output = tf.reshape(x_transposed, [-1, h, w, c])
            return output

    #实现shuffle unit
    def _shufflenet_unit(self, x, num_groups, name, group_conv_bottleneck=True, num_filters=16, stride=1, batchnorm_enabled=True, fusion='concat'):
        # Paper parameters. If you want to change them feel free to pass them as method parameters.
        activation = tf.nn.relu

        with tf.variable_scope(name) as scope:
            residual = x
            bottleneck_filters = (num_filters // 4) if fusion == 'add' else (num_filters - residual.get_shape()[
                3].value) // 4

            #GC+shuffle
            if group_conv_bottleneck:
                bottleneck = self._grouped_conv2d(x=x, name='Gbottleneck', k_size=1, num_filters=bottleneck_filters, num_groups=num_groups,
                                            pad='VALID',activation=activation, batchnorm_enabled=batchnorm_enabled)
                shuffled = self._channel_shuffle(bottleneck, num_groups,name='channel_shuffle')
            else:
                bottleneck = self._grouped_conv2d(x=x, name='Gbottleneck', k_size=1, num_filters=bottleneck_filters, num_groups=num_groups,
                                            pad='VALID',activation=activation, batchnorm_enabled=batchnorm_enabled)
                shuffled = bottleneck

            #DWC
            padded = tf.pad(shuffled, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
            depthwise = self._depthwise_conv2d(x=padded,name='depthwise', k_size=3, stride=stride, pad='VALID',activation=None, batchnorm_enabled=batchnorm_enabled)
            
            #Avgpooling
            if stride == 2:
                residual_pooled = self.avgpooling(residual, kernel_size=3, stride=stride, padding='SAME')
            else:
                residual_pooled = residual

            #GC+concat
            if fusion == 'concat':
                group_conv1x1 = self._grouped_conv2d(x=depthwise,name='Gconv1x1',
                                            k_size=1,
                                            num_filters=num_filters - residual.get_shape()[3].value,
                                            num_groups=num_groups,
                                            pad='VALID',
                                            activation=None,
                                            batchnorm_enabled=batchnorm_enabled)
                return activation(tf.concat([residual_pooled, group_conv1x1], axis=-1))
            elif fusion == 'add':
                group_conv1x1 = self._grouped_conv2d(x=depthwise,name='Gconv1x1',
                                            k_size=1,
                                            num_filters=num_filters,
                                            num_groups=num_groups,
                                            pad='VALID',
                                            activation=None,
                                            batchnorm_enabled=batchnorm_enabled)
                #_debug(group_conv1x1)
                residual_match = residual_pooled
                # This is used if the number of filters of the residual block is different from that
                # of the group convolution.
                if num_filters != residual_pooled.get_shape()[3].value:
                    residual_match = self.conv2d(inputdata=residual_pooled, out_channel=num_filters,
                                            kernel_size=1,
                                            stride=1,
                                            use_bias=False,
                                            padding='VALID', 
                                            name = 'residual_match')

                return activation(group_conv1x1 + residual_match)
            else:
                raise ValueError("Specify whether the fusion is \'concat\' or \'add\'")

    #实现Stage
    def stage(self, x, num_groups=3, stage=2, repeat=3, batchnorm_enabled=True):

        if 2 <= stage <= 4:
            stage_layer = self._shufflenet_unit(
                                            x=x,
                                            num_groups=num_groups,
                                            name='stage' + str(stage) + '_0', 
                                            group_conv_bottleneck=not (stage == 2),
                                            num_filters=
                                            self.output_channels[str(num_groups)][stage - 2],
                                            stride=2,
                                            fusion='concat',
                                            batchnorm_enabled=batchnorm_enabled)
            for i in range(1, repeat + 1):
                stage_layer = self._shufflenet_unit(
                                              x=stage_layer,
                                              num_groups=num_groups,
                                              name='stage' + str(stage) + '_' + str(i),
                                              group_conv_bottleneck=True,
                                              num_filters=
                                              self.output_channels[str(num_groups)][stage - 2],
                                              stride=1,
                                              fusion='add',
                                              batchnorm_enabled=batchnorm_enabled)
            return stage_layer
        else:
            raise ValueError("Stage should be from 2 -> 4")

    #实现shuffleNet
    def encode(self, input_tensor, name):

        ret = OrderedDict()
        with tf.variable_scope(name):

            #Conv1+BN+relu
            conv1 = self._conv_stage(input_tensor=input_tensor, out_dims=self.output_channels['conv1'],
                                k_size=3,
                                pad='VALID',
                                stride=2,
                                name='conv1')

            #Maxpooling
            padded = tf.pad(conv1, [[0, 0], [0, 1], [0, 1], [0, 0]], "CONSTANT")
            max_pool = self.maxpooling(inputdata=padded, kernel_size=3, stride=2, padding='VALID',name='max_pool')

            stage2 = self.stage(max_pool, stage=2, repeat=3)
            ret['stage2'] = dict()
            ret['stage2']['data'] = stage2
            ret['stage2']['shape'] = stage2.get_shape().as_list()

            stage3 = self.stage(stage2, stage=3, repeat=7)
            ret['stage3'] = dict()
            ret['stage3']['data'] = stage3
            ret['stage3']['shape'] = stage3.get_shape().as_list()
            stage4 = self.stage(stage3, stage=4, repeat=3)


            # First Experiment is to use the regular conv2d
#            score_fr = self.conv2d(inputdata=stage4, out_channel=self.num_classes, kernel_size=1, stride=1,
#                               use_bias=False, padding='SAME', name='conv_1c_1x1')

            ret['stage4'] = dict()
            ret['stage4']['data'] = stage4
            ret['stage4']['shape'] = stage4.get_shape().as_list()

            print("\nEncoder ShuffleNet is built successfully\n\n")
            return ret

if __name__ == '__main__':
    a = tf.placeholder(dtype=tf.float32, shape=[1, 640, 480, 24], name='input')
    encoder = ShuffleEncoder(phase=tf.constant('train', dtype=tf.string))
    print('gc')
    ret = encoder._grouped_conv2d(a,'gc',k_size=1)
    print(ret.shape)
    print('sc')
    ret3 = encoder._channel_shuffle(ret, 3, 'sc')
    print(ret3.shape)
    print('dwc')
    padded = tf.pad(ret3, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
    ret2 = encoder._depthwise_conv2d(padded,'dwc',pad='VALID')
    print(ret2.shape)
    print('gc2')
    ret = encoder._grouped_conv2d(ret3,'gc2',k_size=1,pad='VALID')
    print(ret.shape)

    print('su')
    ret4 = encoder._shufflenet_unit(a, num_groups=3, name='su',num_filters=240)
    print(ret4.shape)
    print('stage')
    ret5 = encoder.stage(a)
    print(ret5.shape)
    print('encode')
    ret6 = encoder.encode(a,'encode')
    for layer_name, layer_info in ret6.items():
        print('layer name: {:s} shape: {}'.format(layer_name, layer_info['shape']))

#    ret = encoder.encode(a, name='encode')
#    for layer_name, layer_info in ret.items():
#        print('layer name: {:s} shape: {}'.format(layer_name, layer_info['shape']))

