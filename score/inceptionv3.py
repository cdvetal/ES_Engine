"""
from score.scoreinterface import ScoreInterface
import tensorflow.keras as keras


class Scoring(ScoreInterface):
    def __init__(self):
        super(Scoring, self).__init__()
        self.model = keras.applications.inception_v3.InceptionV3()

    def predict(self, batch, explain=False):
        return self.model.predict(batch)

    def get_target_size(self):
        return (299, 299)

    def get_input_preprocessor(self):
        return keras.applications.inception_v3.preprocess_input
"""


from torchvision.models import inception_v3, Inception_V3_Weights

from score.scoreinterface import ScoreInterface


class Scoring(ScoreInterface):
    def __init__(self):
        super(Scoring, self).__init__()
        weights = inception_v3.DEFAULT
        self.model = Inception_V3_Weights(weights=weights)
        self.model.eval()

        self.preprocess = weights.transforms()

    def predict(self, batch, explain=False):
        return self.model(batch)

    def get_target_size(self):
        return (299, 299)

    def get_input_preprocessor(self):
        return self.preprocess

