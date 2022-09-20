class RenderingInterface:
    model = None

    def __init__(self, args):
        self.args = args
        self.genotype_size = 0
        self.header_length = 1

    def load_model(self, config, checkpoint):
        pass

    def render(self, a, img_size, cur_iteration):
        pass

