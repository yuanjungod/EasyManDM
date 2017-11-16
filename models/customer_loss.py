# -*- coding:utf-8 -*-
from keras.layers import Dense
import numpy as np
from keras.utils import np_utils
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from model_base import ModelBase
from keras.metrics import categorical_accuracy
from keras.preprocessing import sequence


class CustomerLoss(ModelBase):
    def __init__(self, num_class):
        super(CustomerLoss, self).__init__()
        self.num_classes = num_class
        self.model.add(Conv2D(128, (2, 2), padding='same', input_shape=(3, 97, 1), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(1, 2)))
        self.model.add(Conv2D(64, (2, 3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(1, 2)))
        self.model.add(Conv2D(32, (2, 5), padding='valid', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(15, (1, 5), padding='valid', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(1, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(self.num_classes, activation='softmax'))
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        adadelta = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
        self.model.compile(loss='categorical_crossentropy', optimizer=adadelta, metrics=[categorical_accuracy])



if __name__ == "__main__":
    CustomerLoss(2)
