from metrics import *
import tensorflow as tf
from tensorflow import keras
import os


def base_modelarcface(n_classes, **kwargs):
    '''

    :param n_classes:
    :param kwargs:
    :return model: arcface+Densetnet121:
    '''
    base1 = tf.keras.applications.DenseNet121(include_top=False,
                                              input_shape=(160, 160,3),
                                              weights='imagenet')
    y = tf.keras.layers.Input(n_classes, )
    x = tf.keras.layers.Flatten()(base1.output)
    x = tf.keras.layers.Dense(512)(x)
    output = ArcFace(n_classes, regularizer=tf.keras.regularizers.l2(1e-4))([x, y])

    model = tf.keras.models.Model([base1.input, y], output)

    return model

def base_modelsphereface(n_classes, **kwargs):

    '''

    :param n_classes:
    :param kwargs:
    :return sphereface+mobilenetv2:
    '''
    base1 = tf.keras.applications.MobileNetV2(include_top=False,
                                              input_shape=(160, 160, 3),
                                              weights='imagenet')
    y = tf.keras.layers.Input(n_classes, )
    x = tf.keras.layers.Flatten()(base1.output)
    x = tf.keras.layers.Dense(512)(x)
    output = SphereFace(n_classes, regularizer=tf.keras.regularizers.l2(1e-4))([x, y])

    model = tf.keras.models.Model([base1.input, y], output)

    return model


def base_modelscosface(n_classes, **kwargs):
    '''

    :param n_classes:
    :param kwargs:
    :return cosface+resnet:
    '''
    base1 = tf.keras.applications.ResNet101(include_top=False,
                                            input_shape=(160, 160, 3),
                                            weights='imagenet')
    y = tf.keras.layers.Input(n_classes)
    x = tf.keras.layers.Flatten()(base1.output)
    x = tf.keras.layers.Dense(512)(x)
    output = SphereFace(n_classes, regularizer=tf.keras.regularizers.l2(1e-4))([x, y])

    model = tf.keras.models.Model([base1.input, y], output)

    return model

