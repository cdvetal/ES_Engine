"""
from score.scoreinterface import ScoreInterface
from efficientnet.keras import EfficientNetB7
from efficientnet.keras import preprocess_input


class Scoring(ScoreInterface):
    def __init__(self):
        super(Scoring, self).__init__()
        self.model = EfficientNetB7(weights='imagenet')

    def predict(self, batch, explain=False):
        return self.model.predict(batch)

    def get_target_size(self):
        return (600, 600)

    def get_input_preprocessor(self):
        return preprocess_input
"""

from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights

from score.scoreinterface import ScoreInterface


class Scoring(ScoreInterface):
    def __init__(self):
        super(Scoring, self).__init__()
        weights = EfficientNet_B7_Weights.DEFAULT
        self.model = efficientnet_b7(weights=weights)
        self.model.eval()

        self.preprocess = weights.transforms()

    def predict(self, batch, explain=False):
        return self.model(batch)

    def get_target_size(self):
        return (224, 224)

    def get_input_preprocessor(self):
        return self.preprocess

