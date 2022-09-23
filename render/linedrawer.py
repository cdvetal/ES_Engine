import cairo
import numpy as np
from PIL import Image

from render.renderinterface import RenderingInterface


class LineRenderer(RenderingInterface):
    def __init__(self, args):
        super(LineRenderer, self).__init__(args)

        self.num_lines = args.num_lines
        self.img_size = args.img_size

        self.stroke_length = 8

        self.genotype_size = ((self.stroke_length * 6) + 3)
        self.real_genotype_size = self.genotype_size * self.num_lines

    def chunks(self, array):
        img = np.array(array)
        return np.reshape(img, (self.num_lines, self.genotype_size))

    def __str__(self):
        return "linedrawer"

    def bound(self, value, low, high):
        return max(low, min(high, value))

    def render(self, a, cur_iteration):
        ims = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.img_size, self.img_size)
        cr = cairo.Context(ims)

        max_width = 2 * self.img_size / 100
        min_width = 0.5 * self.img_size / 100

        # background shape
        cr.set_source_rgba(242/255.0, 238/255.0, 203/255.0, 1.0)  # everything on cairo appears to be between 0 and 1
        cr.rectangle(0, 0, self.img_size, self.img_size)  # define a rectangle and then fill it
        cr.fill()

        for e in a:
            num_segments = self.stroke_length
            radius = 0.5
            radius_x = 0.5  # radius * canvas_height / canvas_width
            p0 = (0.5 + radius_x * (e[0] - 0.5), 0.5 + radius * (e[1] - 0.5))

            w = map_number(e[3], 0, 1, min_width, max_width)

            cr.set_source_rgb(0, 0, 0)
            # line width
            cr.set_line_width(w)

            ind = 3
            for j in range(num_segments):
                radius = 1.0 / (num_segments + 2)
                p1 = (p0[0] + radius_x * (e[ind] - 0.5), p0[1] + radius * (e[ind + 1] - 0.5))
                p2 = (p1[0] + radius_x * (e[ind + 2] - 0.5), p1[1] + radius * (e[ind + 3] - 0.5))
                p3 = (p2[0] + radius_x * (e[ind + 4] - 0.5), p2[1] + radius * (e[ind + 5] - 0.5))

                ind += 6

                cr.move_to(p0[0] * self.img_size, p0[1] * self.img_size)
                cr.curve_to(p1[0] * self.img_size, p1[1] * self.img_size, p2[0] * self.img_size, p2[1] * self.img_size, p3[0] * self.img_size,
                            p3[1] * self.img_size)

                cr.stroke()

                p0 = (self.bound(p3[0], 0, 1), self.bound(p3[1], 0, 1))

        pilMode = 'RGB'
        # argbArray = numpy.fromstring( ims.get_data(), 'c' ).reshape( -1, 4 )
        argbArray = np.fromstring(bytes(ims.get_data()), 'c').reshape(-1, 4)
        rgbArray = argbArray[:, 2::-1]
        pilData = rgbArray.reshape(-1).tostring()
        pilImage = Image.frombuffer(pilMode,
                                    (ims.get_width(), ims.get_height()), pilData, "raw",
                                    pilMode, 0, 1)
        pilImage = pilImage.convert('RGB')

        # draw line with round line caps (circles at the end)
        # draw.polygon(
        #   ((x1,y1), (x2,y2), (x3,y3)), (R,G,B), outline=None)
        return pilImage
