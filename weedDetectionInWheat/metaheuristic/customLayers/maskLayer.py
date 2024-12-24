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
        self.mask = np.array(mask)
    
    def build(self, inputShape):

        self.maskVariable = self.add_weight(
            shape=self.mask.shape,
            initializer=tf.constant_initializer(self.mask),
            trainable=False,
            name="mask_variable",
        )   

    def call(self, inputs):
        return inputs * self.maskVariable
    
    def set_mask(self, new_mask):
        self.maskVariable.assign(new_mask)

    def get_config(self):
        config = super(MaskLayer, self).get_config()
        config.update({"mask": self.mask.tolist()})
        return config
    
    @classmethod
    def from_config(cls, config):
        mask = tf.convert_to_tensor(config['mask'], dtype=tf.float32)
        return cls(mask)