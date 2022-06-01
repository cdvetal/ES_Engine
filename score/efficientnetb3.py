from scoreinterface import ScoreInterface
import os
from efficientnet.tfkeras import EfficientNetB3
from efficientnet.tfkeras import preprocess_input


class Scoring(ScoreInterface):
    def __init__(self, config):
        super(Scoring, self).__init__()
        self.model = EfficientNetB3(weights='imagenet')

    def predict(self, batch, explain=False):
        return self.model.predict(batch)

    def get_target_size(self):
        return (300, 300)

    def get_input_preprocessor(self):
        return preprocess_input

