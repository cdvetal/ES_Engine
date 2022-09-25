"""
from score.scoreinterface import ScoreInterface
from efficientnet.tfkeras import EfficientNetB0
from efficientnet.tfkeras import preprocess_input


class Scoring(ScoreInterface):
    def __init__(self):
        super(Scoring, self).__init__()
        self.model = EfficientNetB0(weights='imagenet')

    def predict(self, batch, explain=False):
        return self.model.predict(batch)

    def get_target_size(self):
        return (224, 224)

    def get_input_preprocessor(self):
        return preprocess_input
"""

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from score.scoreinterface import ScoreInterface


class Scoring(ScoreInterface):
    def __init__(self):
        super(Scoring, self).__init__()
        weights = EfficientNet_B0_Weights.DEFAULT
        self.model = efficientnet_b0(weights=weights)
        self.model.eval()

        self.preprocess = weights.transforms()

    def predict(self, batch, explain=False):
        return self.model(batch)

    def get_target_size(self):
        return (224, 224)

    def get_input_preprocessor(self):
        return self.preprocess

