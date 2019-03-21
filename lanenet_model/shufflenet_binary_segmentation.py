#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-11 上午11:33
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : lanenet_binary_segmentation.py
# @IDE: PyCharm Community Edition
"""
实现LaneNet中的二分类图像分割模型
"""
import tensorflow as tf

from encoder_decoder_model import vgg_encoder
from encoder_decoder_model import fcn_decoder
from encoder_decoder_model import dense_encoder
from encoder_decoder_model import cnn_basenet
from encoder_decoder_model import shuffle_encoder


class ShuffleNetBinarySeg(cnn_basenet.CNNBaseModel):
    """
    实现语义分割模型
    """
    def __init__(self, phase, net_flag='shuffle'):
        """
        """
        super(ShuffleNetBinarySeg, self).__init__()
        self._net_flag = net_flag
        self._phase = phase
        if self._net_flag == 'vgg':
            self._encoder = vgg_encoder.VGG16Encoder(phase=phase)
        elif self._net_flag == 'dense':
            self._encoder = dense_encoder.DenseEncoder(l=20, growthrate=8,
                                                       with_bc=True,
                                                       phase=self._phase,
                                                       n=5)
        elif self._net_flag == 'shuffle':
            self._encoder = shuffle_encoder.ShuffleEncoder(phase=phase)
        self._decoder = fcn_decoder.FCNDecoder(phase=phase)
        return

    def __str__(self):
        """

        :return:
        """
        info = 'Semantic Segmentation use {:s} as basenet to encode'.format(self._net_flag)
        return info

    def build_model(self, input_tensor, name):
        """
        前向传播过程
        :param input_tensor:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            # first encode
            encode_ret = self._encoder.encode(input_tensor=input_tensor,
                                              name='encode')

            # second decode
            if self._net_flag.lower() == 'vgg':
                decode_ret = self._decoder.decode(input_tensor_dict=encode_ret,
                                                  name='decode',
                                                  decode_layer_list=['pool5',
                                                                     'pool4',
                                                                     'pool3'])
                return decode_ret
            elif self._net_flag.lower() == 'dense':
                decode_ret = self._decoder.decode(input_tensor_dict=encode_ret,
                                                  name='decode',
                                                  decode_layer_list=['Dense_Block_5',
                                                                     'Dense_Block_4',
                                                                     'Dense_Block_3'])
                return decode_ret
            elif self._net_flag.lower() == 'shuffle':
                decode_ret = self._decoder.decode(input_tensor_dict=encode_ret,
                                                  name='decode',
                                                  decode_layer_list=['stage4',
                                                                     'stage3',
                                                                     'stage2'])
                return decode_ret

    def compute_loss(self, input_tensor, label, name):
        """
        计算损失函数
        :param input_tensor:
        :param label:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            # 前向传播获取logits
            inference_ret = self.build_model(input_tensor=input_tensor, name='inference')
            # 计算损失
            decode_logits = inference_ret['logits']
            # 加入bounded inverse class weights
            inverse_class_weights = tf.divide(1.0,
                                              tf.log(tf.add(tf.constant(1.02, tf.float32),
                                                            tf.nn.softmax(decode_logits))))

            decode_logits_weighted = tf.multiply(decode_logits, inverse_class_weights)
            '''第一步是对网络最后一层的输出做一个softmax，这一步通常是求取输出属于某一类的概率'''
            '''第二步是softmax的输出向量[Y1，Y2，Y3，...]和样本的实际标签做一个交叉熵'''
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=decode_logits_weighted, labels=tf.squeeze(label, squeeze_dims=[3]),
                name='entropy_loss')
            loss = tf.reduce_mean(loss)

            ret = dict()
            ret['entropy_loss'] = loss
            ret['inference_logits'] = inference_ret['logits']

            return ret

    def inference(self, input_tensor, name):
        """

        :param input_tensor:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            # 前向传播获取logits
            inference_ret = self.build_model(input_tensor=input_tensor, name='inference')
            # 计算二值分割损失函数,相当于把最后两维度合并成一维吧
            decode_logits = inference_ret['logits']
            binary_seg_ret = tf.nn.softmax(logits=decode_logits)
            binary_seg_ret = tf.argmax(binary_seg_ret, axis=-1)

            return binary_seg_ret


if __name__ == '__main__':
    model = ShuffleNetBinarySeg(tf.constant('train', dtype=tf.string))
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input')
    label = tf.placeholder(dtype=tf.int32, shape=[1, 256, 512, 1], name='label')
    loss = model.compute_loss(input_tensor=input_tensor, label=label, name='loss')
    print(loss['entropy_loss'].get_shape().as_list())
    print(loss['entropy_loss'])
