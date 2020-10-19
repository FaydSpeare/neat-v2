import tensorflow as tf
import numpy as np

def from_hyper_net(net, cppn_inputs, model):
    x = np.array(net.think(cppn_inputs)[0])
    x /= np.max(abs(x))
    weight_matrix = 3 * (x > 0.2) * x
    dense = model.layers[1]
    dense.kernel = tf.constant(weight_matrix.astype('float32'))


