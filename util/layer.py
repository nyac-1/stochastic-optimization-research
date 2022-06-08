from __future__ import division
import random
import math
import numpy as np
import tensorflow as tf

sigmoid = lambda z: float(1/(1 + np.exp(-z)))
relu = lambda z: 0 if(z<=0) else float(z)
standard = lambda z: z

class Layer:
    def __init__(self,name,weights, bias, X,activation):
        self.name = name
        self.weights = weights
        self.bias = bias
        self.n_x = weights.shape[1]
        self.n_y = weights.shape[0]
        self.n_params = self.n_x * self.n_y + self.n_y
        self.X = X # Input from previous layer
        self.activation = activation
    
    def forward_pass(self):
         return np.dot(self.weights, self.X) + self.bias
    
    def apply_activation(self):
        forward_values = self.forward_pass()
        return np.vectorize(self.activation)(forward_values)
    
def bin_cost(values, y):
    return tf.nn.sigmoid_cross_entropy_with_logits(labels= y, logits= values).numpy().mean()

def reg_cost(values, y):
    return tf.keras.metrics.mean_squared_error(y_true=values, y_pred=y).numpy()[0]

def unpack_ff_weights(weights, input_shape, mid_layer_shape):
    end = input_shape[0]*mid_layer_shape[0]
    a = weights[0:end].reshape(mid_layer_shape[0],input_shape[0])
    
    b = weights[end:end+mid_layer_shape[0]].reshape(mid_layer_shape[0],1)
    end+=mid_layer_shape[0]
    
    c = weights[end:end+mid_layer_shape[0]].reshape(1,mid_layer_shape[0])
    end+=mid_layer_shape[0]
    
    d = weights[end:end+1].reshape(1,1)
    
    return (a,b,c,d)

def pack_ff_weights(weights_1, bias_1, weights_2, bias_2):
    return np.concatenate((weights_1.flatten(), bias_1.flatten(), weights_2.flatten(), bias_2.flatten()), axis=0)

def calculate_dimensions(input_size, computation_layer, output_layer):
    
    temp= input_size * computation_layer
    temp+= computation_layer
    temp+= (output_layer*computation_layer)
    temp+= output_layer
    
    return temp

def forward_pass_weights_bin(weights, X_train, y_train,input_size, computational_layer, output_layer):
    
    weights_1, bias_1, weights_2, bias_2 = unpack_ff_weights(weights,(input_size,computational_layer),(computational_layer,output_layer))
    layerOne = Layer("input->hidden",weights_1,bias_1,X_train,relu)
    A = layerOne.apply_activation()
    layerTwo = Layer("hidden->output",weights_2,bias_2,A,sigmoid)
    output = layerTwo.apply_activation()
    cost = bin_cost(output, y_train)
    return cost, output
    
def forward_pass_weights_reg(weights,X_train, y_train,input_size, computational_layer, output_layer):
    
    weights_1, bias_1, weights_2, bias_2 = unpack_ff_weights(weights,(input_size,computational_layer),(computational_layer,output_layer))
    layerOne = Layer("input->hidden",weights_1,bias_1,X_train,relu)
    A = layerOne.apply_activation()
    layerTwo = Layer("hidden->output",weights_2,bias_2,A,standard)
    output = layerTwo.apply_activation()
    cost = reg_cost(y_train, output)
    return cost, output

def mae(values, y):
    return tf.keras.metrics.mean_absolute_error(y_true=values, y_pred=y).numpy()[0]

