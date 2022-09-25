"""
from score.scoreinterface import ScoringInterface
import tensorflow.keras as keras

class Scoring(ScoringInterface): 
    def __init__(self):
        super(Scoring, self).__init__()
        self.model = keras.applications.densenet.DenseNet169()

    def predict(self, batch, explain=False):
        return self.model.predict(batch)

    def get_target_size(self):
        return (224, 224)

    def get_input_preprocessor(self):
        return keras.applications.densenet.preprocess_input
"""
from torchvision.models import DenseNet169_Weights, densenet169

from score.scoreinterface import ScoreInterface


class Scoring(ScoreInterface):
    def __init__(self):
        super(Scoring, self).__init__()
        weights = DenseNet169_Weights.DEFAULT
        self.model = densenet169(weights=weights)
        self.model.eval()

        self.preprocess = weights.transforms()

    def predict(self, batch, explain=False):
        return self.model(batch)

    def get_target_size(self):
        return (224, 224)

    def get_input_preprocessor(self):
        return self.preprocess
