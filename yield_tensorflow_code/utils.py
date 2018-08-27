# -*- coding: utf-8 -*- 

import tensorflow as tf 

def deconv_length(dim_size, stride_size, kernel_size, padding):
	if dim_size is None:
		raise ValueError('dim_size is None.')
	if padding == 'VALID':
		dim_size = dim_size * stride_size + max(kernel_size - stride_size, 0)
	elif padding == 'SAME':
		dim_size = dim_size * stride_size 
	return dim_size 

def get_size(box):
	if isinstance(box, int):
		return (box, ) * 2
	else:
		return (box[0], box[1]) 

def padding2d(input_, padding=(2, 2)):
	padding = get_size(padding)
	# padding: ((top, bottom), (left, right))
	zt, zb = get_size(padding[0])
	zl, zr = get_size(padding[1])
	x_shape = input_.get_shape().as_list()

	zero_pad_l = tf.zeros(shape=[x_shape[0], zl, x_shape[-1]])
	zero_pad_r = tf.zeros(shape=[x_shape[0], zr, x_shape[-1]])
	zero_pad_t = tf.zeros(shape=[zt, x_shape[1] + zl + zr, x_shape[-1]])
	zero_pad_b = tf.zeros(shape=[zb, x_shape[1] + zl + zr, x_shape[-1]])

	output_ = tf.concat([zero_pad_l, input_], axis=1)  # padding left 
	output_ = tf.concat([output_, zero_pad_r], axis=1)  # padding right 
	output_ = tf.concat([zero_pad_t, output_], axis=0)  # padding top 
	output_ = tf.concat([output_, zero_pad_b], axis=0) # padding bottom 
	return output_
