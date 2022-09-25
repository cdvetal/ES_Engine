"""
from score.scoreinterface import ScoreInterface
from efficientnet.keras import EfficientNetB6
from efficientnet.keras import preprocess_input


class Scoring(ScoreInterface):
    def __init__(self):
        super(Scoring, self).__init__()
        self.model = EfficientNetB6(weights='imagenet')

    def predict(self, batch, explain=False):
        return self.model.predict(batch)

    def get_target_size(self):
        return (528, 528)

    def get_input_preprocessor(self):
        return preprocess_input
"""

from torchvision.models import efficientnet_b6, EfficientNet_B6_Weights

from score.scoreinterface import ScoreInterface


class Scoring(ScoreInterface):
    def __init__(self):
        super(Scoring, self).__init__()
        weights = EfficientNet_B6_Weights.DEFAULT
        self.model = efficientnet_b6(weights=weights)
        self.model.eval()

        self.preprocess = weights.transforms()

    def predict(self, batch, explain=False):
        return self.model(batch)

    def get_target_size(self):
        return (224, 224)

    def get_input_preprocessor(self):
        return self.preprocess

