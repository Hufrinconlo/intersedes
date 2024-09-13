"""
====================
Dice Cofficient Loss
====================

.. math:: -2\\frac{|\mathcal{M} \otimes \hat{\mathcal{M}}|}{|\mathcal{M}| - |\hat{\mathcal{M}}|}

 
.. [1] `Random Fourier Features-Based Deep Learning Improvement with Class Activation Interpretability for Nerve Structure Segmentation`_

.. [2] `Multi categorical Dice loss?`_

.. [3] `RFF-Nerve-UTP`_


.. _`Random Fourier Features-Based Deep Learning Improvement with Class Activation Interpretability for Nerve Structure Segmentation`: http://www.sdss.org/dr14/help/glossary/#stripe

.. _`Multi categorical Dice loss?`: https://stats.stackexchange.com/questions/285640/multi-categorical-dice-loss

.. _`RFF-Nerve-UTP`: https://github.com/cralji/RFF-Nerve-UTP
"""

import tensorflow as tf
from keras.losses import Loss
from keras import backend as K
from keras.utils import to_categorical


class DiceCoefficient(Loss):
    def __init__(self, smooth=1., target_class=None, name='DiceCoefficient', **kwargs):
        self.smooth = smooth
        self.target_class = target_class
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        # Create a mask to exclude the value 2
        mask = tf.logical_and(tf.not_equal(y_true, 2), tf.not_equal(y_pred, 2))
        y_true = tf.where(mask, y_true, tf.zeros_like(y_true))
        y_pred = tf.where(mask, y_pred, tf.zeros_like(y_pred))

        intersection = K.sum(y_true * y_pred, axis=[1, 2])
        union = K.sum(y_true, axis=[1, 2]) + K.sum(y_pred, axis=[1, 2])
        dice_coef = -(2. * intersection + self.smooth) / (union + self.smooth)

        if self.target_class is not None:
            dice_coef = tf.gather(dice_coef, self.target_class, axis=1)
        else:
            dice_coef = K.mean(dice_coef, axis=-1)

        return dice_coef

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "smooth": self.smooth, "target_class": self.target_class}


class SparseCategoricalDiceCoefficient(DiceCoefficient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        y_true = to_categorical(y_true)
        return super().call(y_true, y_pred)
