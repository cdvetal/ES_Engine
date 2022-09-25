"""
from score.scoreinterface import ScoreInterface
from tensorflow import keras

class Scoring(ScoreInterface):
    def __init__(self):
        super(Scoring, self).__init__()
        self.model = keras.applications.densenet.DenseNet121()

    def predict(self, batch, explain=False):
        return self.model.predict(batch)

    def get_target_size(self):
        return (224, 224)

    def get_input_preprocessor(self):
        return keras.applications.densenet.preprocess_input
"""

from score.scoreinterface import ScoreInterface
from torchvision.models import densenet121, DenseNet121_Weights


class Scoring(ScoreInterface):
    def __init__(self):
        super(Scoring, self).__init__()
        weights = DenseNet121_Weights.DEFAULT
        self.model = densenet121(weights=weights)
        self.model.eval()

        self.preprocess = weights.transforms()

    def predict(self, batch, explain=False):
        return self.model(batch)

    def get_target_size(self):
        return (224, 224)

    def get_input_preprocessor(self):
        return self.preprocess
