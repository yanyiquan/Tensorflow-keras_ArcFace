import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from metrics import *
from model import *
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from keras.preprocessing.image import ImageDataGenerator
import cv2
import os

def process_data(path, **kwargs):

  '''
  read image and process
  '''

  X, Y = []
  for file in os.listdir(path):
    for file_name in os.listdir(os.path.join(path, file)):
      img = cv2.imread(os.path.join(path, file, file_name))
      img = cv2.resize(img, (160, 160))
      X.append(img)
      Y.append(file)
  return np.asarray(X), np.asarray(Y)

def _most_similarity(embed_vecs, vec, labels):
  sim = cosine_similarity(embed_vecs, vec)
  sim = np.squeeze(sim, axis = 1)
  argmax = np.argsort(sim)[::-1][:-1]
  label = [labels[idx] for idx in argmax][0]
  return label


def test_model(x_train, y_train, x_test, y_test):
  y_preds = []
  for vec in x_test:
    vec = vec.reshape(1, -1)
    y_pred = _most_similarity(x_train, vec, y_train)
    y_preds.append(y_pred)

    return accuracy_score(y_preds, y_test)


if __name__=='__main__':
    path = './face1'
    X, Y = process_data(path)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=30)
    print("[INFO] Example for train: {:.2f}".format(len(y_train)))
    print("[INFO] Example for test: {:.2f}".format(len(y_test)))
    x_train = x_train/255
    x_test = x_test/255
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')
    y_train = tf.keras.utils.to_categorical(y_train, 8)
    y_test = tf.keras.utils.to_categorical(y_test, 8)
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)
    callbacks = [tf.keras.callbacks.Callback.EarlyStopping(monitor='val_acc', patience=5),
                 tf.keras.callbacks.Callback.ModelCheckpoint('model_checkpoint',
                                                             monitor='val_acc',
                                                             verbose=0,
                                                             save_best_only=True,
                                                             save_weights_only=True),
                 tf.keras.callbacks.Callback.TensorBoard(log_dir='logs'),
                 tf.keras.callbacks.Callback.ReduceLROnPlateau(monitor='val_acc',
                                                               factor=0.1,
                                                               patience=5,
                                                               verbose=0)]

    model1 = base_modelarcface(8)
    model1.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
                   loss=tf.keras.losses.categorical_crossentropy,
                   metrics=['accuracy'])
    model1.fit([x_train, y_train], y_train, steps_per_epoch=len(x_train)/32,
               epochs=20, batch_size=32, validation_data=([x_test, y_test], y_test),
               verbose=1)
    model1.save('./model/arcface.h5')

    model2 = base_modelsphereface(8)
    model2.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
                   loss=tf.keras.losses.categorical_crossentropy,
                   metrics=['acc'])
    model2.fit([x_train, y_train], y_train, steps_per_epoch=50, epochs=50,
               batch_size=32, validation_data=([x_test, y_test], y_test),
               verbose=1)
    model2.save('./model/sphereface.h5')

    model3 = base_modelscosface(8)
    model3.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
                   loss=tf.keras.losses.categorical_crossentropy,
                   metrics=['accuracy'])
    model3.fit([x_train, y_train], y_train, steps_per_epoch=len(x_train)/32, epochs=50,
               batch_size=32, validation_data=([x_test, y_test], y_test),
               verbose=1)
    model3.save('./model/cosface.h5')


    model1 = load_model('./model/arcface.h5')
    model_arcface = tf.keras.models.Model(inputs=[model1.input[0]], outputs=[model1.layers[-3].output])
    output_arcface = model_arcface.predict(x_test, verbose=1)
    output_arcface /= np.linalg.norm(output_arcface, axis=1, keep_dim=True)
    x_train_arc = model_arcface.predict(x_train, verbose=1)

    model2 = load_model('./model/sphereface.h5')
    model_sphereface = tf.keras.models.Model(inputs=[model2.input[0]], outputs=[model2.layers[-3].output])
    output_sphereface = model_sphereface.predict(x_test, verbose = 1)
    output_sphereface /= np.linalg.norm(output_sphereface, axis=1, keep_dim=True)
    x_train_sphere = model_sphereface.predict(x_train, verbose=1)

    model3 = load_model('./model/cosface.h5')
    model_cosface = tf.keras.models.Model(inputs=[model3.input[0]], outputs=[model3.layers[-3].output])
    output_cosface = model_cosface.predict(x_test, verbose=1)
    output_cosface /= np.linalg.norm(output_cosface, axis=1, keep_dim=True)
    x_train_cos = model_cosface.predict(x_train, verbose=1)

    output_feature = np.mean(output_arcface + output_cosface + output_sphereface, axis=0)# mean of arcface, sphereface and cosface
    x_train_mean = np.mean(x_train_arc, x_train_sphere, x_train_cos, axis=0)

    print("[INFO] Accuracy arcface: {:.2f}".format(test_model(x_train_arc, y_train, output_arcface, y_test)))
    print("[INFO] Accuracy sphereface: {:.2f}".format(test_model(x_train_sphere, y_train, output_sphereface, y_test)))
    print("[INFO] Accuracy cosface: {:.2f}".format(test_model(x_train_cos, y_train, output_cosface, y_test)))
    print("[INFO] Accuracy mean feature: {:.2f}".format(test_model(x_train_mean, y_train, output_feature, y_test)))