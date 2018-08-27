# -*- coding: utf-8 -*- 

import tensorflow as tf 
import keras.backend.tensorflow_backend as KBT 
from keras.models import load_model 
from keras.layers import Dense, Conv2D, Conv2DTranspose, BatchNormalization 

import json 
import numpy as np 
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
KBT.set_session(tf.Session(config=config))

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

def get_model_config():
    """Write the current model config 
    """
    try:
        model_config = {}
        model_config['model_name'] = model_name
        model_config = json.dumps(model_config)
        with open('config.json', 'w') as file:
            file.write(model_config)
            
    except Exception as e:
        raise e 

def get_model_weights():
    """ Load a current model weights.
    """
    try:
        model_weights = {}
        for layer in model.layers:
            layer.trainable = False 
            model_weights[layer.name] = layer.get_weights()
        # Save model weights 
        model_weights_name = model_name + '.npy'
        np.save(model_weights_name, model_weights)
        
    except Exception as e:
        raise e
        
def gene_model():
    """ Generate model.py based on configuration of model 
    """
    model_config = json.loads(model.to_json())# json.loads: json -> dict 
    layers = model_config["config"]["layers"]

    try:
        file = open('model.py', 'w') 
        file.write("# -*- coding : utf-8 -*-\n")
        file.write("import tensorflow as tf \nfrom ops import * \n \n\n")
        file.write('class %s:\n\tdef __init__(self): \n' % model_name)
        file.write("\t\tself.model_name = '%s'\n" % model_name)
        file.write("\t\tself.inference()\n\n")
        file.write('\tdef inference(self): \n')

        for layer in layers:
            if layer['class_name'] == 'InputLayer':
                config = layer['config']
                name = config['name']
                shape = config['batch_input_shape']
                shape = [str(e) for e in shape]
                shape = '[' + ', '.join(shape) + ']'
                dtype = 'tf.' + config['dtype'] 
                scope = name 
                name = name.replace('/', '_')
                code_seq = "\t\tself.%s = tf.placeholder(dtype=%s, shape=%s, name='%s')\n" % (name, dtype, shape, scope)
                file.write(code_seq) 

            elif layer['class_name'] == 'Reshape':
                config = layer['config']
                name = config['name']
                target_shape = config['target_shape']
                inbound = 'self.' + layer['inbound_nodes'][0][0][0].replace('/', '_')
                scope = name
                name = name.replace('/', '_')
                code_seq = "\t\tself.%s = reshape(%s, target_shape=%s, scope='%s')\n" % (name, inbound, target_shape, scope) 
                file.write(code_seq)
            
            elif layer['class_name'] == 'Flatten':
                config = layer['config']
                name = config['name'] 
                inbound = 'self.' + layer['inbound_nodes'][0][0][0].replace('/', '_')
                scope = name
                name = name.replace('/', '_')
                code_seq = "\t\tself.%s = flatten(%s, scope='%s')\n" % (name, inbound, scope)
                file.write(code_seq)

            elif layer['class_name'] == 'Dense':
                config = layer['config']
                name = config['name']
                units = config['units']
                use_bias = config['use_bias']
                activation = config['activation']
                inbound = 'self.' + layer['inbound_nodes'][0][0][0].replace('/', '_')
                scope = name
                name = name.replace('/', '_')
                code_seq = "\t\tself.%s = dense(%s, units=%s, use_bias=%s, activation='%s', scope='%s')\n" % (name, inbound, units, use_bias, activation, scope)
                file.write(code_seq)

            elif layer['class_name'] == 'Add':
                config = layer['config']
                name = config['name']
                inbounds = []
                for e in layer['inbound_nodes'][0]:
                    inbounds.append('self.' + e[0].replace('/', '_'))
                inbounds = '[' + ', '.join(inbounds) + ']'
                scope = name
                name = name.replace('/', '_')
                code_seq = "\t\tself.%s = add(%s, scope='%s')\n" % (name, inbounds, scope)
                file.write(code_seq)

            elif layer["class_name"] == "Conv2D": 
                config = layer['config'] 
                name = config['name'] 
                inbounds = "self." + layer['inbound_nodes'][0][0][0].replace('/', '_')
                filters = config['filters'] 
                k_size = config['kernel_size'] 
                s_size = config['strides'] 
                activation = config['activation']
                use_bias = config['use_bias']
                padding = config['padding']
                scope = name
                name = name.replace('/', '_')
                code_seq = "\t\tself.%s = conv2d(%s, filters=%s, k_size=%s, s_size=%s, activation='%s', use_bias=%s, padding='%s', scope='%s')\n" % (name, inbounds, filters, k_size, s_size, activation, use_bias, padding.upper(), scope)
                file.write(code_seq)

            elif layer['class_name'] == 'BatchNormalization':
                config = layer['config']
                name = config['name']
                
                momentum = config['momentum']
                epsilon = config['epsilon']
                center = config['center']
                scale = config['scale']
                
                inbounds = "self." + layer['inbound_nodes'][0][0][0].replace('/', '_')
                if 'training' in layer['inbound_nodes'][0][0][-1]:
                    training = layer['inbound_nodes'][0][0][-1]['training'] 
                else:
                    training = False 
                scope = name
                name = name.replace('/', '_')
                code_seq = "\t\tself.%s = batch_norm(%s, momentum=%s, epsilon=%s, scale=%s, center=%s, training=%s, scope='%s')\n" % (name, inbounds, momentum, epsilon, scale, center, training, scope) 
                file.write(code_seq)

            elif layer['class_name'] == 'Activation':
                config = layer['config']
                inbounds = "self." + layer['inbound_nodes'][0][0][0].replace('/', '_')
                name = config['name']
                scope = name 
                name = name.replace('/', '_')
                if config['activation'] == 'relu':
                    code_seq = "\t\tself.%s = relu(%s, scope='%s')\n" % (name, inbounds, scope)
                file.write(code_seq)

            elif layer['class_name'] == 'LeakyReLU':
                config = layer['config']
                inbound = 'self.' + layer['inbound_nodes'][0][0][0].replace('/', '_')
                name = config['name']
                leak = config['alpha']
                scope = name 
                name = name.replace('/', '_')
                code_seq = "\t\tself.%s = lrelu(%s, leak=%s, scope='%s')\n" % (name, inbound, leak, scope)
                file.write(code_seq)

            elif layer['class_name'] == 'Concatenate':
                config = layer['config']
                inbounds = []
                for e in layer['inbound_nodes'][0]: 
                    inbounds.append('self.' + e[0].replace('/', '_'))
                inbounds = '[' + ', '.join(inbounds) + ']'
                name = config['name']
                axis = config['axis']
                scope = name
                name = name.replace('/', '_')
                code_seq = "\t\tself.%s = concatenate(%s, axis=%s, scope='%s')\n" % (name, inbounds, axis, scope)
                file.write(code_seq)

            elif layer['class_name'] == 'Conv2DTranspose':
                config = layer['config']
                name = config['name']
                filters = config['filters']
                kernel_size = config['kernel_size']
                strides = config['strides']
                padding = config['padding']
                use_bias = config['use_bias']
                inbounds = "self." + layer['inbound_nodes'][0][0][0].replace('/', '_')
                scope = name
                name = name.replace('/', '_')
                code_seq = "\t\tself.%s = deconv2d(%s, filters=%s, k_size=%s, s_size=%s, use_bias=%s, padding='%s', scope='%s')\n" % (name, inbounds, filters, kernel_size, strides, use_bias, padding.upper(), scope)
                file.write(code_seq)

            elif layer['class_name'] == 'UpSampling2D':
                config = layer['config']
                name = config['name']
                size = config['size']
                inbound = 'self.' + layer['inbound_nodes'][0][0][0].replace('/', '_')
                scope = name
                name = name.replace('/', '_')
                code_seq = "\t\tself.%s = up_sampling2d(%s, size=%s, scope='%s')\n" % (name, inbound, size, scope)
                file.write(code_seq)

            elif layer['class_name'] == 'MaxPooling2D':
                config = layer['config']
                name = config['name']
                pool_size = config['pool_size']
                padding = config['padding']
                strides = config['strides']
                inbound = 'self.' + layer['inbound_nodes'][0][0][0].replace('/', '_')
                scope = name
                name = name.replace('/', '_')
                code_seq = "\t\tself.%s = max_pooling2d(%s, k_size=%s, s_size=%s, padding='%s', scope='%s')\n" % (name, inbound, pool_size, strides, padding.upper(), scope)
                file.write(code_seq)

            elif layer['class_name'] == 'AveragePooling2D':
                config = layer['config']
                name = config['name']
                pool_size = config['pool_size']
                padding = config['padding']
                strides = config['strides']
                inbound = 'self.' + layer['inbound_nodes'][0][0][0].replace('/', '_')
                scope = name
                name = name.replace('/', '_')
                code_seq = "\t\tself.%s = avg_pooling2d(%s, k_size=%s, s_size=%s, padding='%s', scope='%s')\n" % (name, inbound, pool_size, strides, padding.upper(), scope)
                file.write(code_seq)
            
            elif layer['class_name'] == 'GlobalAveragePooling2D':
                config = layer['config']
                name = config['name']
                inbound = 'self.' + layer['inbound_nodes'][0][0][0].replace('/', '_')
                scope = name
                name = name.replace('/', '_')
                code_seq = "\t\tself.%s = global_average_pooling(%s, scope='%s')\n" % (name, inbound, scope)
                file.write(code_seq)

            elif layer['class_name'] == 'ZeroPadding2D':
                config = layer['config']
                name = config['name']
                padding = config["padding"]
                inbound = 'self.' + layer['inbound_nodes'][0][0][0].replace('/', '_')
                scope = name
                name = name.replace('/', '_')
                code_seq = "\t\tself.%s = zero_padding2d(%s, padding=%s, scope='%s')\n" % (name, inbound, padding, scope)
                file.write(code_seq)
            else:
                raise Exception('Cons_model.py has not such %s layer.' % layer['class_name'])
            
        print('Have generated model.py')
            
    except Exception as e:
        raise e 
    
    finally:
        file.close()
        

def main(argv):
    get_model_config() 
    get_model_weights() 
    gene_model() 
    
if __name__ == '__main__':
    tf.app.run() 
    
    
