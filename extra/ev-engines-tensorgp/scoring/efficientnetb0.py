from classloader import ScoringInterface
import os
from efficientnet.tfkeras import EfficientNetB0
from efficientnet.tfkeras import preprocess_input


class Scoring(ScoringInterface): 
    def __init__(self, config):
        super(Scoring, self).__init__()
        self.model = EfficientNetB0(weights='imagenet')

    def predict(self, batch, explain=False):
        return self.model.predict(batch)

    def get_target_size(self):
        return (224, 224)

    def get_input_preprocessor(self):
        return preprocess_input

