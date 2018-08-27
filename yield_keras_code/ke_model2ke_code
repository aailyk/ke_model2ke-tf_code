# -*- coding: utf-8 -*- 

import tensorflow as tf 
import keras.backend.tensorflow_backend as KBT 
from keras.models import load_model 
from keras.layers import InputLayer, Dense, Conv2D, Conv2DTranspose, BatchNormalization, ZeroPadding2D 
from keras.initializers import VarianceScaling, Zeros, Ones 

import json 
import numpy as np 
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# KBT.set_session(tf.Session(config=config))

flags = tf.app.flags 
flags.DEFINE_string('model_addr', '', "The source address. ['']")
flags.DEFINE_string('model_name', '', "The source model name. ['']")
FLAGS = flags.FLAGS 

print(FLAGS)
''' shell
python cons_model.py --model_addr '' --model_name line_head.net 
'''

# To load model
print('Load model begin! ...') 
model = load_model(os.path.join(FLAGS.model_addr, FLAGS.model_name)) 
print('... Load model completed!') 

model_name = os.path.splitext(FLAGS.model_name)[0]
        
def gene_model():
    """ Generate model.py based on configuration of model 
    """
    model_config = json.loads(model.to_json())# json.loads: json -> dict 
    layers = model_config["config"]["layers"]

    try:
        file = open('model.py', 'w') 
        file.write("# -*- coding : utf-8 -*-\n")
        file.write("from keras.models import * \nfrom keras.layers import * \nfrom keras.initializers import * \n\n")
        file.write('class %s:\n\tdef __init__(self): \n' % model_name)
        file.write("\t\tself.model_name = '%s'\n" % model_name) 
        file.write("\t\tself.inference()\n\n")
        file.write('\tdef inference(self): \n')

        for layer in layers:
            if layer['class_name'] == 'InputLayer':
                config = layer['config']
                name = config['name']
                scope = name
                name = name.replace('/', '_')
                bishp = config['batch_input_shape'][1:]
                dtype = config['dtype']
                sparse = config['sparse']
                code_seq = "\t\tself.%s = Input(shape=%s, dtype='%s', sparse=%s, name='%s')\n" % (name, bishp, dtype, sparse, scope)
                file.write(code_seq) 

            elif layer['class_name'] == 'Reshape':
                config = layer['config']
                name = config['name']
                scope = name
                name = name.replace('/', '_')
                trainable = config['trainable'] 
                target_shape = config['target_shape']
                inbound = 'self.' + layer['inbound_nodes'][0][0][0].replace('/', '_')
                code_seq = "\t\tself.%s = Reshape(trainable=%s, target_shape=%s, name='%s')(%s)\n" % (name, trainable, target_shape, scope, inbound)
                file.write(code_seq)
            
            elif layer['class_name'] == 'Flatten':
                config = layer['config']
                name = config['name'] 
                scope = name
                name = name.replace('/', '_')
                trainable = config['trainable']
                inbound = 'self.' + layer['inbound_nodes'][0][0][0].replace('/', '_')
                code_seq = "\t\tself.%s = Flatten(trainable=%s, name='%s')(%s)\n" % (name, trainable, scope, inbound)
                file.write(code_seq)

            elif layer['class_name'] == 'Dense':
                config = layer['config']
                name = config['name']
                scope = name
                name = name.replace('/', '_')
                trainable = config['trainable']
                units = config['units']
                activation = config['activation']
                use_bias = config['use_bias']
                k_init = config['kernel_initializer']
                if config['kernel_initializer']['class_name'] == 'VarianceScaling':
                    w_initer = "VarianceScaling(scale=%s, mode='%s', distribution='%s')" % (k_init['config']['scale'], k_init['config']['mode'], k_init['config']['distribution'])
                if config['bias_initializer']['class_name'] == 'Zeros':
                    b_initer = 'zeros'
                inbound = 'self.' + layer['inbound_nodes'][0][0][0].replace('/', '_')
                code_seq = "\t\tself.%s = Dense(trainable=%s, units=%s, activation='%s', use_bias=%s, kernel_initializer=%s, bias_initializer='%s', name='%s')(%s)\n" % (name, trainable, units, activation, use_bias, w_initer, b_initer, scope, inbound)
                file.write(code_seq)

            elif layer['class_name'] == 'Add':
                config = layer['config']
                name = config['name']
                scope = name
                name = name.replace('/', '_')
                trainable = config['trainable']
                inbounds = []
                for e in layer['inbound_nodes'][0]:
                    inbounds.append('self.' + e[0].replace('/', '_'))
                inbounds = '[' + ', '.join(inbounds) + ']'
                code_seq = "\t\tself.%s = Add(trainable=%s, name='%s')(%s)\n" % (name, trainable, scope, inbounds)
                file.write(code_seq)

            elif layer["class_name"] == "Conv2D": 
                config = layer['config'] 
                name = config['name']
                scope = name 
                name = name.replace('/', '_')
                trainable = config['trainable'] 
                filters = config['filters']
                k_size = config['kernel_size']
                s_size = config['strides'] 
                padding = config['padding'] 
                dila_rate = config['dilation_rate'] 
                activation = config['activation']
                use_bias = config['use_bias'] 
                k_init = config['kernel_initializer']
                b_init = config['bias_initializer'] 
                
                if k_init['class_name'] == 'VarianceScaling':
                    w_initer = "VarianceScaling(scale=%s, mode='%s', distribution='%s')" % (k_init['config']['scale'], k_init['config']['mode'], k_init['config']['distribution'])
                if b_init['class_name'] == 'Zeros':
                    b_initer = 'zeros' 
                    
                inbounds = "self." + layer['inbound_nodes'][0][0][0].replace('/', '_')
                
                code_seq = "\t\tself.%s = Conv2D(trainable=%s, filters=%s, kernel_size=%s, strides=%s, padding='%s', dilation_rate=%s , activation='%s', use_bias=%s, kernel_initializer=%s, bias_initializer='%s', name='%s')(%s)\n" % (name, trainable, filters, k_size, s_size, padding, dila_rate, activation, use_bias, w_initer, b_initer, scope, inbounds)
                file.write(code_seq)

            elif layer['class_name'] == 'BatchNormalization':
                config = layer['config']
                name = config['name']
                scope = name
                name = name.replace('/', '_') 
                trainable = config['trainable']
                axis = config['axis']
                momentum = config['momentum']
                epsilon = config['epsilon']
                center = config['center']
                scale = config['scale']
                beta_init = config['beta_initializer']['class_name'].lower()
                gamma_init = config['gamma_initializer']['class_name'].lower()
                mm_init = config['moving_mean_initializer']['class_name'].lower()
                mv_init = config['moving_variance_initializer']['class_name'].lower()
                inbounds = "self." + layer['inbound_nodes'][0][0][0].replace('/', '_')
                if 'training' in layer['inbound_nodes'][0][0][-1]:
                    training = layer['inbound_nodes'][0][0][-1]['training'] 
                else:
                    training = False 
                code_seq = "\t\tself.%s = BatchNormalization(trainable=%s, axis=%s, momentum=%s, epsilon=%s, center=%s, scale=%s, beta_initializer='%s', gamma_initializer='%s', moving_mean_initializer='%s', moving_variance_initializer='%s', name='%s')(%s, training=%s)\n" % (name, trainable, axis, momentum, epsilon, center, scale, beta_init, gamma_init, mm_init, mv_init, scope, inbounds, training)
                file.write(code_seq)

            elif layer['class_name'] == 'Activation':
                config = layer['config']
                inbounds = "self." + layer['inbound_nodes'][0][0][0].replace('/', '_')
                name = config['name']
                scope = name 
                name = name.replace('/', '_')
                trainable = config['trainable']
                activation = config['activation']
                code_seq = "\t\tself.%s = Activation(trainable=%s, activation='%s', name='%s')(%s)\n" % (name, trainable, activation, scope, inbounds)
                file.write(code_seq)

            elif layer['class_name'] == 'LeakyReLU':
                config = layer['config']
                name = config['name']
                scope = name 
                name = name.replace('/', '_')
                trainable = config['trainable']
                alpha = config['alpha']
                inbound = 'self.' + layer['inbound_nodes'][0][0][0].replace('/', '_')
                code_seq = "\t\tself.%s = LeakyReLU(trainable=%s, alpha=%s, name='%s')(%s)\n" % (name, trainable, alpha, scope, inbound)
                file.write(code_seq)

            elif layer['class_name'] == 'Concatenate':
                config = layer['config']
                name = config['name']
                scope = name
                name = name.replace('/', '_')
                trainable = config['trainable']
                axis = config['axis']
                inbounds = []
                for e in layer['inbound_nodes'][0]: 
                    inbounds.append('self.' + e[0].replace('/', '_'))
                inbounds = '[' + ', '.join(inbounds) + ']'
                code_seq = "\t\tself.%s = Concatenate(trainable=%s, axis=%s, name='%s')(%s)\n" % (name, trainable, axis, scope, inbounds)
                file.write(code_seq)

            elif layer['class_name'] == 'Conv2DTranspose':
                config = layer['config']
                name = config['name']
                scope = name
                name = name.replace('/', '_')
                trainable = config['trainable']
                filters = config['filters']
                kernel_size = config['kernel_size']
                strides = config['strides']
                padding = config['padding']
                activation = config['activation']
                use_bias = config['use_bias']
                k_init = config['kernel_initializer']
                b_init = config['bias_initializer'] 
                
                if k_init['class_name'] == 'VarianceScaling':
                    w_initer = "VarianceScaling(scale=%s, mode='%s', distribution='%s')" % (k_init['config']['scale'], k_init['config']['mode'], k_init['config']['distribution'])
                if b_init['class_name'] == 'Zeros':
                    b_initer = 'zeros'
                
                inbounds = "self." + layer['inbound_nodes'][0][0][0].replace('/', '_')
                code_seq = "\t\tself.%s = Conv2DTranspose(trainable=%s, filters=%s, kernel_size=%s, strides=%s, padding='%s', activation='%s', use_bias=%s, kernel_initializer=%s, bias_initializer='%s', name='%s')(%s)\n" % (name, trainable, filters, kernel_size, strides, padding, activation, use_bias, w_initer, b_initer, scope, inbounds)
                file.write(code_seq)

            elif layer['class_name'] == 'UpSampling2D':
                config = layer['config']
                name = config['name']
                scope = name
                name = name.replace('/', '_')
                trainable = config['trainable']
                size = config['size']
                inbound = 'self.' + layer['inbound_nodes'][0][0][0].replace('/', '_')
                code_seq = "\t\tself.%s = UpSampling2D(trainable=%s, size=%s, name='%s')(%s)\n" % (name, trainable, size, scope, inbound)
                file.write(code_seq)

            elif layer['class_name'] == 'MaxPooling2D':
                config = layer['config']
                name = config['name']
                scope = name
                name = name.replace('/', '_')
                trainable = config['trainable']
                p_size = config['pool_size']
                padding = config['padding']
                strides = config['strides']
                inbound = 'self.' + layer['inbound_nodes'][0][0][0].replace('/', '_')
                code_seq = "\t\tself.%s = MaxPooling2D(trainable=%s, pool_size=%s, padding='%s', strides=%s)(%s)\n" % (name, trainable, p_size, padding, strides, inbound)
                file.write(code_seq)

            elif layer['class_name'] == 'AveragePooling2D':
                config = layer['config']
                name = config['name']
                scope = name
                name = name.replace('/', '_')
                trainable = config['trainable']
                pool_size = config['pool_size']
                padding = config['padding']
                strides = config['strides']
                inbound = 'self.' + layer['inbound_nodes'][0][0][0].replace('/', '_')
                code_seq = "\t\tself.%s = AveragePooling2D(trainable=%s, pool_size=%s, padding='%s', strides=%s, name='%s')(%s)\n" % (name, trainable, pool_size, padding, strides, scope, inbound) 
                file.write(code_seq)
            
            elif layer['class_name'] == 'GlobalAveragePooling2D':
                config = layer['config']
                name = config['name']
                scope = name
                name = name.replace('/', '_')
                trainable = config['trainable']
                inbound = 'self.' + layer['inbound_nodes'][0][0][0].replace('/', '_')
                code_seq = "\t\tself.%s = GlobalAveragePooling2D(trainable=%s, name='%s')(%s)\n" % (name, trainable, scope, inbound)
                file.write(code_seq)

            elif layer['class_name'] == 'ZeroPadding2D':
                config = layer['config']
                name = config['name']
                scope = name 
                name = name.replace('/', '_')
                trainable = config['trainable']
                padding = config['padding'] 
                inbound = 'self.' + layer['inbound_nodes'][0][0][0].replace('/', '_')
                code_seq = "\t\tself.%s = ZeroPadding2D(trainable=%s, padding=%s, name='%s')(%s)\n" % (name, trainable, padding, scope, inbound)
                file.write(code_seq)
            else:
                raise Exception('Cons_model.py has not such %s layer.' % layer['class_name'])
            
        print('Have generated model.py')
            
    except Exception as e:
        raise e 
    
    finally:
        file.close()
        

def main(argv):
    gene_model() 
    
if __name__ == '__main__':
    tf.app.run() 
    
    
