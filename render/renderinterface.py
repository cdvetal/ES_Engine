class RenderingInterface:
    model = None

    def __init__(self):
        self.genotype_size = 0

    def load_model(self, config, checkpoint):
        pass

    def render(self, a, img_size):
        pass

