from classloader import ScoringInterface
import os
from efficientnet.keras import EfficientNetB7
from efficientnet.keras import preprocess_input


class Scoring(ScoringInterface): 
    def __init__(self, config):
        super(Scoring, self).__init__()
        self.model = EfficientNetB7(weights='imagenet')

    def predict(self, batch, explain=False):
        return self.model.predict(batch)

    def get_target_size(self):
        return (600, 600)

    def get_input_preprocessor(self):
        return preprocess_input

