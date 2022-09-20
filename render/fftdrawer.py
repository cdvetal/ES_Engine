import numpy as np

from render.renderinterface import RenderingInterface


# from aphantasia.image import to_valid_rgb, fft_image, pixel_image


def pixel_image(shape, sd, resume):
    pass


def fft_image(shape, sd, decay_power, resume):
    pass


class FFTRenderer(RenderingInterface):
    def __init__(self, args, fft_use="fft"):
        super(FFTRenderer, self).__init__(args)

        shape = [1, 3, args.IMG_SIZE, args.IMG_SIZE]

        resume = None

        if fft_use == "pixel":
            params, image_f, sz = pixel_image(shape, sd=1, resume=resume)
        elif fft_use == "fft":
            params, image_f, sz = fft_image(shape, sd=0.01, decay_power=self.decay, resume=resume)
        else:
            raise ValueError(f"fft drawer does not know how to apply fft_use={fft_use}")

        self.params = params
        # self.image_f = to_valid_rgb(image_f, colors=1.5)

        self.genotype_size = 12
        self.real_genotype_size = self.genotype_size * args.num_lines

    def chunks(self, array):
        img = np.array(array)
        return np.reshape(img, (self.args.num_lines, self.genotype_size))

    def __str__(self):
        return "fftdrawer"

    def render(self, a, img_size, cur_iteration):
        pass

