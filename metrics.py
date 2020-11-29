import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import cv2
import os


class ArcFace(tf.keras.layers.Layer):

    def __init__(self, n_classes, s=30, m=0.5, regularizer=None, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = tf.keras.regularizers.get(regularizer)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'n_classes': self.n_classes
        })
        return config

    def build(self, input_shape):
        super(ArcFace, self).build(input_shape[0])
        self.w = self.add_weight(name="W",
                               shape=(input_shape[0][-1], self.n_classes),
                               initializer='glorot_uniform',
                               trainable=True,
                               regularizer=self.regularizer)

    def call(self, inputs):
        x, y = inputs
        c = K.shape(x)[-1]
        x = tf.math.l2_normalize(x, axis=1)
        w = tf.math.l2_normalize(self.w, axis=0)

        logits = x @ w

        theta = tf.acos(K.clip(logits, -1+K.epsilon(), 1-K.epsilon()))

        target_logits = tf.cos(theta+self.m)

        logits = logits*(1-y)+target_logits*y
        logits *= self.s

        out = tf.nn.softmax(logits)

        return out


class SphereFace(tf.keras.layers.Layer):

    def __init__(self, n_classes, s=30, m=1.25, regularizer=None, **kwargs):
        super(SphereFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = tf.keras.regularizers.get(regularizer)
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'n_classes': self.n_classes
        })
        return config
    def build(self, input_shape):
        super(SphereFace, self).build(input_shape[0])
        self.w = self.add_weight(name="W",
                                 shape=(input_shape[0][-1], self.n_classes),
                                 initializer='glorot_uniform',
                                 trainable=True,
                                 regularizer=self.regularizer)

    def call(self, inputs):
        x, y = inputs
        c = K.shape(x)[-1]
        x = tf.math.l2_normalize(x, axis=1)
        w = tf.math.l2_normalize(self.w, axis=0)

        logits = x @ w

        theta = tf.acos(K.clip(logits, -1 + K.epsilon(), 1 - K.epsilon()))

        target_logits = tf.cos(theta + self.m)

        logits = logits * (1 - y) + target_logits * y
        logits *= self.s

        out = tf.nn.softmax(logits)

        return out


class CosFace(tf.keras.layers.Layer):

    def __init__(self, n_classes, s=30, m=0.35, regularizer=None, **kwargs):
        super(CosFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = tf.keras.regularizers.get(regularizer)

    def build(self, input_shape):
        super(CosFace, self).build(input_shape[0])
        self.w = self.add_weight(name="W",
                                 shape=(input_shape[0][-1], self.n_classes),
                                 initializer="glorot_uniform",
                                 trainable=True,
                                 regularizer=self.regularizer)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'n_classes': self.n_classes
        })
        return config

    def call(self, inputs):
        x, y = inputs

        x = tf.math.l2_normalize(x, axis=1)
        w = tf.math.l2_normalize(self.w, axis=0)

        logits = x @ w

        target_logits = logits - self.m

        logits = logits*(1-y)+target_logits*y

        logits *= self.s

        out = tf.nn.softmax(logits)

        return out








