class RenderingInterface:
    def __init__(self, args):
        pass

    def generate_individual(self):
        pass

    def get_individual(self):
        pass

    def to_adam(self, individual, gradients=True):
        pass

    def chunks(self, array):
        pass

    def __str__(self):
        pass

    def render(self):
        pass

