import math

import cairo
import numpy as np
from PIL import Image

from renderinterface import RenderingInterface
from utils import map_number


class ClipDrawRenderer(RenderingInterface):
    def __init__(self, args):
        super(ClipDrawRenderer, self).__init__(args)

        self.genotype_size = 12
        self.real_genotype_size = self.genotype_size * args.num_lines

    def chunks(self, array):
        img = np.array(array)
        return np.reshape(img, (self.args.num_lines, self.genotype_size))

    def __str__(self):
        return "clipdraw"

    def render(self, a, img_size):
        # split input array into header and rest
        head = a[:self.header_length]
        rest = a[self.header_length:]

        # determine background color from header
        R = head[0][0]
        G = head[0][1]
        B = head[0][2]

        # create the image and drawing context
        # im = Image.new('RGB', (size, size), (R, G, B))
        # draw = ImageDraw.Draw(im, 'RGB')

        ims = cairo.ImageSurface(cairo.FORMAT_ARGB32, img_size, img_size)
        cr = cairo.Context(ims)

        cr.set_source_rgba(R, G, B, 1.0)  # everythingon cairo appears to be between 0 and 1
        cr.rectangle(0, 0, img_size, img_size)  # define a recatangle and then fill it
        cr.fill()

        cr.set_line_cap(cairo.LINE_CAP_ROUND)
        cr.set_line_join(cairo.LINE_JOIN_ROUND)

        radius = 0.1

        max_width = 2.0 * img_size / 100
        min_width = 0.5 * img_size / 100

        for e in rest:
            R = e[0]
            G = e[1]
            B = e[2]

            w = map_number(e[3], 0, 1, min_width, max_width)

            cr.set_source_rgb(R, G, B)
            # line width
            cr.set_line_width(w)

            p0 = (e[4], e[5])
            p1 = (p0[0] + radius * (e[6] - 0.5), p0[1] + radius * (e[7] - 0.5))
            p2 = (p1[0] + radius * (e[8] - 0.5), p1[1] + radius * (e[9] - 0.5))
            p3 = (p2[0] + radius * (e[10] - 0.5), p2[1] + radius * (e[11] - 0.5))

            cr.move_to(p0[0] * img_size, p0[1] * img_size)
            cr.curve_to(p1[0] * img_size, p1[1] * img_size, p2[0] * img_size, p2[1] * img_size, p3[0] * img_size, p3[1] * img_size)

            cr.stroke()

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
