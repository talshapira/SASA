from keras import backend as K
from keras.layers import Layer, multiply
import numpy as np

class ScaledDotProductAttention(Layer):
    """
        Implementation according to:
            "Attention is all you need" by A Vaswani, N Shazeer, N Parmar (2017)
    """

    def __init__(self, return_attention=False, **kwargs):    
        self._return_attention = return_attention
        self.supports_masking = True
        super(ScaledDotProductAttention, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
        self._validate_input_shape(input_shape)

        if not self._return_attention:
            return input_shape[-1]
        else:
            return [input_shape[-1], [input_shape[0][0], input_shape[0][1], input_shape[1][2]]]
    
    def _validate_input_shape(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("Layer received an input shape {0} but expected three inputs (Q, K, V).".format(input_shape))
        else:
            if input_shape[0][0] != input_shape[1][0] or input_shape[1][0] != input_shape[2][0]:
                raise ValueError("All three inputs (Q, V, K) have to have the same batch size; received batch sizes: {0}, {1}, {2}".format(input_shape[0][0], input_shape[1][0], input_shape[2][0]))
            if input_shape[0][1] != input_shape[1][1] or input_shape[1][1] != input_shape[2][1]:
                raise ValueError("All three inputs (Q, V, K) have to have the same length; received lengths: {0}, {1}, {2}".format(input_shape[0][0], input_shape[1][0], input_shape[2][0]))
            if input_shape[0][2] != input_shape[1][2]:
                raise ValueError("Input shapes of Q {0} and V {1} do not match.".format(input_shape[0], input_shape[1]))
    
    def build(self, input_shape):
        self._validate_input_shape(input_shape)
        
        super(ScaledDotProductAttention, self).build(input_shape)
    
    def call(self, x, mask=None):
        q, k, v = x
        d_k = q.shape.as_list()[2]

        # in pure tensorflow:
        # weights = tf.matmul(x_batch, tf.transpose(y_batch, perm=[0, 2, 1]))
        # normalized_weights = tf.nn.softmax(weights/scaling)
        # output = tf.matmul(normalized_weights, x_batch)
        
        weights = K.batch_dot(q,  k, axes=[2, 2])

        if mask is not None:
            # add mask weights
            if isinstance(mask, (list, tuple)):
                if len(mask) > 0:
                    raise ValueError("mask can only be a Tensor or a list of length 1 containing a tensor.")

                mask = mask[0]

            weights += -1e10*(1-mask)

        normalized_weights = K.softmax(weights / np.sqrt(d_k))
        output = K.batch_dot(normalized_weights, v)
        
        if self._return_attention:
            return [output, normalized_weights]
        else:
            return output

    def get_config(self):
        config = {'return_attention': self._return_attention}
        base_config = super(ScaledDotProductAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
   

class SourceAwareAttention(Layer):

    def __init__(self, return_attention=False, **kwargs):    
        self._return_attention = return_attention
        self.supports_masking = True
        super(SourceAwareAttention, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
        self._validate_input_shape(input_shape)

        if not self._return_attention:
            return input_shape[-1]
        else:
            return [input_shape[-1], 
                    [input_shape[0][0], input_shape[0][1], input_shape[1][2]], 
                    [input_shape[0][0], input_shape[0][1], input_shape[1][2]]]
    
    def _validate_input_shape(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError("Layer received an input shape {0} but expected four inputs (Q, K, V, S).".format(input_shape))
        else:
            if input_shape[0][0] != input_shape[1][0] or input_shape[1][0] != input_shape[2][0] or input_shape[2][0] != input_shape[3][0]:
                raise ValueError("All three inputs (Q, K, V, S) have to have the same batch size; received batch sizes: {0}, {1}, {2}, {3}".format(input_shape[0][0], input_shape[1][0], input_shape[2][0]),input_shape[3][0])
            if input_shape[0][1] != input_shape[1][1] or input_shape[1][1] != input_shape[2][1] or input_shape[2][1] != input_shape[3][1]:
                raise ValueError("All three inputs (Q, K, V, S) have to have the same length; received lengths: {0}, {1}, {2}, {3}".format(input_shape[0][0], input_shape[1][0], input_shape[2][0]), input_shape[3][0])
            if input_shape[0][2] != input_shape[1][2]:
                raise ValueError("Input shapes of Q {0} and K {1} do not match.".format(input_shape[0], input_shape[1]))
    
    def build(self, input_shape):
        self._validate_input_shape(input_shape)
        
        super(SourceAwareAttention, self).build(input_shape)
    
    def call(self, x, mask=None):
        q, k, v, s = x
        d_k = q.shape.as_list()[2]

        # in pure tensorflow:
        # weights = tf.matmul(x_batch, tf.transpose(y_batch, perm=[0, 2, 1]))
        # normalized_weights = tf.nn.softmax(weights/scaling)
        # output = tf.matmul(normalized_weights, x_batch)
        
        weights = K.batch_dot(q,  k, axes=[2, 2])
        # print("weights", weights.shape)

        if mask is not None:
            # add mask weights
            if isinstance(mask, (list, tuple)):
                if len(mask) > 0:
                    raise ValueError("mask can only be a Tensor or a list of length 1 containing a tensor.")

                mask = mask[0]

            weights += -1e10*(1-mask)

        normalized_weights = K.softmax(weights / np.sqrt(d_k))
        # print("normalized_weights", normalized_weights.shape)
        
        # option A
        unceratinty_weights = K.batch_dot(normalized_weights,s) #if s.shape = (T, hidden_size)
        # print("unceratinty_weights", unceratinty_weights.shape)
#         output = K.batch_dot(unceratinty_weights,v)
        output = multiply([unceratinty_weights, v])
        # print("output", output.shape)
        
        if self._return_attention:
            return [output, normalized_weights, unceratinty_weights]
        else:
            return output

    def get_config(self):
        config = {'return_attention': self._return_attention}
        base_config = super(SourceAwareAttention, self).get_config() #super(ScaledDotProductAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
