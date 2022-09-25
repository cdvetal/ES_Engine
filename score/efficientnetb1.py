"""
from score.scoreinterface import ScoreInterface
from efficientnet.tfkeras import EfficientNetB1
from efficientnet.tfkeras import preprocess_input


class Scoring(ScoreInterface):
    def __init__(self):
        super(Scoring, self).__init__()
        self.model = EfficientNetB1(weights='imagenet')

    def predict(self, batch, explain=False):
        return self.model.predict(batch)

    def get_target_size(self):
        return (240, 240)

    def get_input_preprocessor(self):
        return preprocess_input
"""

from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights

from score.scoreinterface import ScoreInterface


class Scoring(ScoreInterface):
    def __init__(self):
        super(Scoring, self).__init__()
        weights = EfficientNet_B1_Weights.DEFAULT
        self.model = efficientnet_b1(weights=weights)
        self.model.eval()

        self.preprocess = weights.transforms()

    def predict(self, batch, explain=False):
        return self.model(batch)

    def get_target_size(self):
        return (224, 224)

    def get_input_preprocessor(self):
        return self.preprocess

