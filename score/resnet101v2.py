"""
from score.scoreinterface import ScoreInterface
import tensorflow.keras as keras


class Scoring(ScoreInterface):
    def __init__(self):
        super(Scoring, self).__init__()
        self.model = keras.applications.resnet_v2.ResNet101V2()

    def predict(self, batch, explain=False):
        return self.model.predict(batch)

    def get_target_size(self):
        return (224, 224)

    def get_input_preprocessor(self):
        return keras.applications.resnet_v2.preprocess_input
"""

from score.scoreinterface import ScoreInterface
from torchvision.models import resnet101, ResNet101_Weights


class Scoring(ScoreInterface):
    def __init__(self):
        super(Scoring, self).__init__()
        weights = ResNet101_Weights.IMAGENET1K_V2
        self.model = resnet101(weights=weights)
        self.model.eval()

        self.preprocess = weights.transforms()

    def predict(self, batch, explain=False):
        return self.model(batch)

    def get_target_size(self):
        return (224, 224)

    def get_input_preprocessor(self):
        return self.preprocess
