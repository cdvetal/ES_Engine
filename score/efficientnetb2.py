from score.scoreinterface import ScoreInterface
from efficientnet.tfkeras import EfficientNetB2
from efficientnet.tfkeras import preprocess_input


class Scoring(ScoreInterface):
    def __init__(self):
        super(Scoring, self).__init__()
        self.model = EfficientNetB2(weights='imagenet')

    def predict(self, batch, explain=False):
        return self.model.predict(batch)

    def get_target_size(self):
        return (260, 260)

    def get_input_preprocessor(self):
        return preprocess_input

