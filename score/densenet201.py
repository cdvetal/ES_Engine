"""
from score.scoreinterface import ScoringInterface
import tensorflow.keras as keras

class Scoring(ScoringInterface): 
    def __init__(self):
        super(Scoring, self).__init__()
        self.model = keras.applications.densenet.DenseNet201()

    def predict(self, batch, explain=False):
        return self.model.predict(batch)

    def get_target_size(self):
        return (224, 224)

    def get_input_preprocessor(self):
        return keras.applications.densenet.preprocess_input
"""

from torchvision.models import DenseNet201_Weights, densenet201

from score.scoreinterface import ScoreInterface


class Scoring(ScoreInterface):
    def __init__(self):
        super(Scoring, self).__init__()
        weights = DenseNet201_Weights.DEFAULT
        self.model = densenet201(weights=weights)
        self.model.eval()

        self.preprocess = weights.transforms()

    def predict(self, batch, explain=False):
        return self.model(batch)

    def get_target_size(self):
        return (224, 224)

    def get_input_preprocessor(self):
        return self.preprocess
