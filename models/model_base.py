# -*- coding:utf-8 -*-

from keras.models import Sequential
from keras.models import load_model


class ModelBase(object):

    def __init__(self):
        self.model = Sequential()

    def fit(self, x, y, batch_size=32, epochs=10, verbose=1, callbacks=None,
            validation_split=0., validation_data=None, shuffle=True,
            class_weight=None, sample_weight=None, initial_epoch=0):
        self.model.fit(x, y, batch_size, epochs, verbose, callbacks,
                       validation_split, validation_data, shuffle,
                       class_weight, sample_weight, initial_epoch)

    def predict_on_batch(self, x):
        return self.model.predict_on_batch(x)

    def get_model_para(self):
        return self.model.get_config()

    def predict(self, x):
        return self.model.predict(x)

    def save_model(self, model_path):
        self.model.save(model_path)

    @classmethod
    def load_model(cls, model_path):
        load_model(model_path)
