import matplotlib.pyplot as plt
import numpy as np
import os
from keras.callbacks import ModelCheckpoint
import time
from keras.utils import to_categorical
import cv2
import tensorflow as tf
import keras.backend as K
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, UpSampling2D, concatenate
from keras.models import Model
from keras.losses import binary_crossentropy, mean_squared_error, categorical_crossentropy
from keras.optimizers import Adam
import matplotlib
from itertools import product
from functools import partial
import scipy.misc

matplotlib.use('Agg')


def w_binary_crossentropy(y_true, y_pred):
    weights = np.array([[1, 1], [1, 1]])
    # tmp = np.ones(y_pred.shape)
    cross_results = K.binary_crossentropy(y_true, y_pred)
    return K.mean(cross_results, axis=1)

# ncce = partial(w_binary_crossentropy, weights=np.array([[1, 1],[1,1]]))

def my_binary_crossentropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)


class SegFCNet:
    def __init__(self, model_path):
        self.model_path = model_path
        if not os.path.exists(model_path):
            os.mkdir(model_path)

    def build_model(self, input_shape):
        inputs = Input(shape=input_shape)

        conv1 = Conv2D(32, (5, 5), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        pool1 = MaxPooling2D((2, 2), strides=(2, 2))(conv1)
        conv2 = Conv2D(64, (5, 5), activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        pool2 = MaxPooling2D((2, 2), strides=(2, 2))(conv2)
        conv3 = Conv2D(128, (5, 5), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        pool3 = MaxPooling2D((2, 2), strides=(2, 2))(conv3)
        conv4 = Conv2D(256, (5, 5), activation='relu', padding='same', kernel_initializer='he_normal')(pool3)

        drop4 = Dropout(0.1)(conv4)

        up5 = Conv2D(128, (5, 5), activation='relu', padding='same', kernel_initializer='he_normal')\
            (UpSampling2D(size=(2, 2))(drop4))
        # merge4 = concatenate([conv2, up4], axis=3)
        conv5 = Conv2D(128, (5, 5), activation='relu', padding='same', kernel_initializer='he_normal')(up5)

        up6 = Conv2D(64, (5, 5), activation='relu', padding='same', kernel_initializer='he_normal')\
            (UpSampling2D(size=(2, 2))(conv5))
        # merge5 = concatenate([conv1, up5], axis=3)
        conv6 = Conv2D(64, (5, 5), activation='relu', padding='same', kernel_initializer='he_normal')(up6)
        up7 = Conv2D(64, (5, 5), activation='relu', padding='same', kernel_initializer='he_normal')\
            (UpSampling2D(size=(2, 2))(conv6))
        conv7 = Conv2D(32, (5, 5), activation='relu', padding='same', kernel_initializer='he_normal')(up7)

        outputs = Conv2D(2, (1, 1), activation='softmax', padding='same', kernel_initializer='he_normal')(conv7)

        model = Model(input=inputs, output=outputs)

        return model

    def train(self, x_data, y_data, lr=0.002, N_epoch=20, batch_size=16, validation_split=0.01):
        image_shape = x_data.shape[1:]
        class_weights = {0: 1, 1: np.sum(y_data == 0) / np.sum(y_data == 1)}
        print("class weight: 1:{:.2f}".format(np.sum(y_data == 0) / np.sum(y_data == 1)))
        y_data = to_categorical(y_data)
        model = self.build_model(image_shape)
        model.summary()

        checkpointer = ModelCheckpoint(filepath=os.path.join(self.model_path, 'seg_best_weights.h5'),
                                       verbose=0, monitor='loss', save_best_only=True)
        optimizer = Adam(lr=lr)
        model.compile(optimizer=optimizer, loss=binary_crossentropy, metrics=['accuracy'])
        hist = model.fit(x_data, y_data, epochs=N_epoch, batch_size=batch_size, verbose=2, shuffle=True,
                       validation_split=validation_split, callbacks=[checkpointer])
        # fig = plt.figure()
        # plt.plot(hist.history['loss'], '-o')
        # # plt.plot(hist.history['val_loss'], 'g-o')
        # # plt.legend(['loss', 'val_loss'])
        # fig.savefig('seg_train_convergence.png', format='png', bbox_inches='tight', dpi=900)
        # plt.close()

        model.save_weights(os.path.join(self.model_path, 'seg_last_weights.h5'), overwrite=True)

    def predict(self, img, model_name='seg_best_weights.h5'):
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=3)

        image_shape = img.shape[1:]
        model = self.build_model(image_shape)
        # model.compile()
        model.load_weights(os.path.join(self.model_path, model_name))

        predict_results = model.predict(img, verbose=1)
        prob_map = np.squeeze(predict_results[:, :, :, 1])

        # fig = plt.figure()
        # plt.imshow(prob_map, cmap='plasma')  # default is 'viridis'
        # plt.colorbar()
        # fig.savefig('seg_prob_map.png', format='png', bbox_inches='tight', dpi=900)
        # plt.close()
        scipy.misc.imsave('outfile.jpg', (prob_map*200).astype(np.uint8))
        return prob_map