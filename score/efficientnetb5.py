from scoreinterface import ScoreInterface

from efficientnet.tfkeras import EfficientNetB5
from efficientnet.tfkeras import preprocess_input


class Scoring(ScoreInterface):
    def __init__(self):
        super(Scoring, self).__init__()
        self.model = EfficientNetB5(weights='imagenet')

    def predict(self, batch, explain=False):
        return self.model.predict(batch)

    def get_target_size(self):
        return (456, 456)

    def get_input_preprocessor(self):
        return preprocess_input

