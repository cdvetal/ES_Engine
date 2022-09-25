"""
from score.scoreinterface import ScoreInterface
import tensorflow.keras as keras


class Scoring(ScoreInterface):
    def __init__(self):
        super(Scoring, self).__init__()
        self.model = keras.applications.nasnet.NASNetMobile()

    def predict(self, batch, explain=False):
        return self.model.predict(batch)

    def get_target_size(self):
        return (224, 224)

    def get_input_preprocessor(self):
        return keras.applications.nasnet.preprocess_input
"""

from torchvision.models import mnasnet1_3, MNASNet1_3_Weights

from score.scoreinterface import ScoreInterface


class Scoring(ScoreInterface):
    def __init__(self):
        super(Scoring, self).__init__()
        weights = MNASNet1_3_Weights.DEFAULT
        self.model = mnasnet1_3(weights=weights)
        self.model.eval()

        self.preprocess = weights.transforms()

    def predict(self, batch, explain=False):
        return self.model(batch)

    def get_target_size(self):
        return (224, 224)

    def get_input_preprocessor(self):
        return self.preprocess


