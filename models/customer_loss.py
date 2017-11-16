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
        self.model.add(Conv2D(32, (2, 3), padding='valid', input_shape=(3, 97, 1), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(20, (1, 3), padding='valid', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(1, 2)))
        self.model.add(Conv2D(15, (1, 3), padding='valid', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(1, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(self.num_classes, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[categorical_accuracy])


if __name__ == "__main__":
    CustomerLoss(2)
