from scoreinterface import ScoreInterface
import os
import tensorflow.keras as keras


class Scoring(ScoreInterface):
    def __init__(self, config):
        super(Scoring, self).__init__()
        self.model = keras.applications.nasnet.NASNetLarge()

    def predict(self, batch, explain=False):
        return self.model.predict(batch)

    def get_target_size(self):
        return (331, 331)

    def get_input_preprocessor(self):
        return keras.applications.nasnet.preprocess_input

