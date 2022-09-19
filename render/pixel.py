import cairo
import numpy as np
from PIL import Image

from renderinterface import RenderingInterface


class PixelRenderer(RenderingInterface):
    def __init__(self, args):
        super(PixelRenderer, self).__init__(args)

        self.num_lines = args.num_lines

        self.genotype_size = 3
        self.real_genotype_size = self.genotype_size * (args.num_lines * args.num_lines)

    def chunks(self, array):
        img = np.array(array)
        return np.reshape(img, ((self.num_lines * self.num_lines), self.genotype_size))

    def __str__(self):
        return "pixel"

    def render(self, a, img_size):
        ims = cairo.ImageSurface(cairo.FORMAT_ARGB32, img_size, img_size)
        cr = cairo.Context(ims)

        cr.set_source_rgba(1.0, 1.0, 1.0, 1.0)  # everything on cairo appears to be between 0 and 1
        cr.rectangle(0, 0, img_size, img_size)  # define a rectangle and then fill it
        cr.fill()

        width = img_size / self.args.num_lines
        x = 0
        y = 0
        for r in a:
            # determine foreground color from header
            R = r[0]
            G = r[1]
            B = r[2]

            cr.set_source_rgb(R, G, B)
            # line width
            cr.rectangle(x, y, width, width)  # define a rectangle and then fill it
            cr.fill()

            x += width
            if x >= img_size:
                x = 0
                y += width

        pilMode = 'RGB'
        # argbArray = numpy.fromstring( ims.get_data(), 'c' ).reshape( -1, 4 )
        argbArray = np.fromstring(bytes(ims.get_data()), 'c').reshape(-1, 4)
        rgbArray = argbArray[:, 2::-1]
        pilData = rgbArray.reshape(-1).tostring()
        pilImage = Image.frombuffer(pilMode,
                                    (ims.get_width(), ims.get_height()), pilData, "raw",
                                    pilMode, 0, 1)
        pilImage = pilImage.convert('RGB')
        # pilImage.show() # mostra a image no preview

        # draw line with round line caps (circles at the end)
        # draw.polygon(
        #   ((x1,y1), (x2,y2), (x3,y3)), (R,G,B), outline=None)
        return pilImage
