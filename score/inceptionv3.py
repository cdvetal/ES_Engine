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

