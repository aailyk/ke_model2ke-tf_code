# -*- coding : utf-8 -*-

import tensorflow as tf 
from tensorflow.contrib.layers import variance_scaling_initializer as xavier

import numpy as np 
import os 
import json 

import utils 


with open('config.json') as file:
	config = json.load(file)
	model_config = json.loads(config)
    
model_weights = np.load(model_config['model_name'] + '.npy', encoding='latin1').item()

def get_size(box):
	if isinstance(box, int):
		return (box) * 2
	else:
		return (box[0], box[1])

def batch_norm(input_, 
				momentum=0.99, 
				epsilon=0.001,
				scale=True,
				center=True,
				training=False, 
				scope='batch_norm'):
	with tf.variable_scope(scope):
		if scale:
			gamma = tf.constant_initializer(model_weights[scope][0]) 
			if center:
				beta = tf.constant_initializer(model_weights[scope][1])
				moving_mean = tf.constant_initializer(model_weights[scope][2])
				moving_variance = tf.constant_initializer(model_weights[scope][3]) 
			else:
				beta = None 
				moving_mean = tf.constant_initializer(model_weights[scope][1])
				moving_variance = tf.constant_initializer(model_weights[scope][2]) 
		else: 
			gamma = None 
			if center:
				beta = tf.constant_initializer(model_weights[scope][0])
				moving_mean = tf.constant_initializer(model_weights[scope][1])
				moving_variance = tf.constant_initializer(model_weights[scope][2]) 
			else:
				beta = None 
				moving_mean = tf.constant_initializer(model_weights[scope][0])
				moving_variance = tf.constant_initializer(model_weights[scope][1])
		return tf.layers.batch_normalization(input_, 
								momentum=momentum, 
								epsilon=epsilon, 
								center=center, 
								scale=scale, 
								gamma_initializer=gamma,
								beta_initializer=beta,
								moving_mean_initializer=moving_mean,
								moving_variance_initializer=moving_variance,
								training=training)

def conv2d(input_, 
			filters, 
			k_size=(5, 5), 
			s_size=(2, 2), 
			use_bias=True,
			padding='SAME', 
			activation='linear', 
			scope="conv2d"):
	k_h, k_w = get_size(k_size)
	s_h, s_w = get_size(s_size)

	with tf.variable_scope(scope):
		w = tf.get_variable("kernel", shape=[k_h, k_w, input_.get_shape()[-1], filters], initializer=tf.constant_initializer(model_weights[scope][0]))
		conv = tf.nn.conv2d(input_, w, strides=[1, s_h, s_w, 1], padding=padding)
		if use_bias:
			bias = tf.get_variable("bias", [filters], initializer=tf.constant_initializer(model_weights[scope][1])) 
			conv = tf.nn.bias_add(conv, bias) 
		if activation == 'linear':
			return conv 
		elif activation == 'relu': 
			return tf.nn.relu(conv)    

def deconv2d(input_, 
			filters, 
			k_size=(5, 5), 
			s_size=(2, 2), 
			use_bias=True, 
			padding='SAME',
			scope="deconv2d"):
	"""
	input_.get_shape: [batch_size, in_height, in_width, in_channel] 
	output_shape: [batch_size, out_height, out_width, out_channel] 
	"""
	k_h, k_w = get_size(k_size)
	s_h, s_w = get_size(s_size)
	input_shape = input_.get_shape().as_list()
	out_height = utils.deconv_length(input_shape[1], s_h, k_h, padding)
	out_width = utils.deconv_length(input_shape[2], s_w, k_w, padding)
	output_shape = [input_shape[0], out_height, out_width, filters]
	with tf.variable_scope(scope):
		w = tf.get_variable('kernel', [k_h, k_w, output_shape[-1], input_shape[-1]], initializer=tf.constant_initializer(model_weights[scope][0]))
		deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, s_h, s_w, 1], padding=padding)
		if use_bias:
			bias = tf.get_variable('bias', [output_shape[-1]], initializer=tf.constant_initializer(model_weights[scope][1]))
			deconv = tf.nn.bias_add(deconv, bias)
		return deconv

def max_pooling2d(input_, k_size=(3, 3), s_size=(2, 2), padding='SAME', scope='max_pooling2d'):
	k_h, k_w = get_size(k_size)
	s_h, s_w = get_size(s_size)
	with tf.name_scope(scope):
		return tf.nn.max_pool(input_, [1, k_h, k_w, 1], [1, s_h, s_w, 1], padding=padding)

def avg_pooling2d(input_, k_size=(3, 3), s_size=(2, 2), padding='SAME', scope='avg_pooling2d'):
	k_h, k_w = get_size(k_size)
	s_h, s_w = get_size(s_size)
	with tf.name_scope(scope):
		return tf.nn.avg_pool(input_, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

def dense(input_, 
			units, 
			use_bias=True,
			activation='linear',
			scope="linear"):
	with tf.variable_scope(scope): 
		mat = tf.get_variable('kernel', shape=[input_.get_shape()[-1], units], initializer=tf.constant_initializer(model_weights[scope][0]))
            
		if use_bias:
			bias = tf.get_variable('bias', [units], initializer=tf.constant_initializer(model_weights[scope][1])) 
			fc = tf.matmul(input_, mat) + bias
		else:
			fc = tf.matmul(input_, mat) 
		if activation == 'linear': 
			return fc 
		elif activation == 'softmax': 
			return tf.nn.softmax(fc)
		elif activation == 'relu': 
			return tf.nn.relu(fc)

def relu(x, scope='relu'):
	with tf.name_scope(scope):
		return tf.nn.relu(x)
    
def lrelu(x, leak=0.10000000149011612, scope="lrelu"):
	with tf.name_scope(scope):
		return tf.maximum(x, leak*x)

def add(x, scope='add'):
	with tf.name_scope(scope):
		return tf.add(*x) # *: 解包
    
def concatenate(x, axis=-1, scope='concatenate'):
	with tf.name_scope(scope):
		return tf.concat(x, axis)

def flatten(x, scope='flatten'):
	with tf.name_scope(scope):
		return tf.layers.flatten(x) 
    
def reshape(input_, target_shape, scope='reshape'):
	target_shape = [-1] + target_shape 
	with tf.name_scope(scope): 
		return tf.reshape(input_, target_shape) 
    
def global_average_pooling(input_, scope='global_average_pooling'):
	with tf.name_scope(scope):
		x1 = tf.reduce_mean(input_, axis=1)
		x2 = tf.reduce_mean(x1, axis=1)
		return x2
    
def up_sampling2d(input_, size=(2, 2), scope='up_sampling2d'):
	hf, wf = utils.get_size(size)    
	original_shape = x.get_shape().as_list()
	with tf.name_scope(scope):
		original_shape = int_shape(x)
		new_shape = tf.shape(x)[1:3]
		new_shape *= tf.constant(np.array([hf, wf]).astype('int32'))
		x = tf.image.resize_nearest_neighbor(x, new_shape)
		x.set_shape((None, original_shape[1]*hf, original_shape[2]*wf, original_shape[-1]))
		return x 

def zero_padding2d(input_, padding=(2,2), scope='zero_padding2d'):
	with tf.name_scope(scope):
		output_ = tf.map_fn(fn=lambda x: utils.padding2d(x, padding), elems=input_, dtype=tf.float32) 
		return output_ 
