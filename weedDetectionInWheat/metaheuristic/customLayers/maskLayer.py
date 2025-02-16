import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.layers import Layer, Flatten, Dense, Input
from tensorflow.keras.models import load_model, Model

@register_keras_serializable(package="Custom", name="MaskLayer")
class MaskLayer(Layer):

    def __init__(self, mask=None, name="mask", **kwargs):
        super(MaskLayer, self).__init__(name=name, **kwargs)
        self.mask = np.array(mask, dtype=np.int32)
        self.feature_dimension = None
    
    def build(self, input_shape):
        self.mask_variable = self.add_weight(
            shape=self.mask.shape,
            initializer=tf.constant_initializer(self.mask),
            trainable=False,
            name="mask_variable",
        )
 
        self.feature_dimension = int(tf.reduce_sum(self.mask).numpy())   

    def call(self, inputs):
        masked_input = inputs * self.mask_variable
        
        mask_boolean = tf.cast(self.mask_variable, tf.bool)
        mask_boolean = tf.reshape(mask_boolean, tf.shape(masked_input))
        masked_input = tf.boolean_mask(masked_input, mask_boolean)

        batch_size = tf.shape(inputs)[0] 
        return tf.reshape(masked_input, [batch_size, self.feature_dimension])
        
    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        return (batch_size, self.feature_dimension)  
    
    def set_mask(self, new_mask):
        self.mask_variable.assign(new_mask)

    def get_config(self):
        config = super(MaskLayer, self).get_config()
        config.update({"mask": self.mask.tolist()})
        return config
    
    def get_weights(self):
        return [self.mask_variable.numpy()]

    def set_weights(self, weights):
        self.mask_variable.assign(weights[0])
        
    @classmethod
    def from_config(cls, config):
        mask = tf.convert_to_tensor(config['mask'], dtype=tf.float32)
        return cls(mask)