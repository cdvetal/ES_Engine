import cairo
import numpy as np
from PIL import Image

from render.renderinterface import RenderingInterface
from utils import map_number


class ClipDrawRenderer(RenderingInterface):
    def __init__(self, args):
        super(ClipDrawRenderer, self).__init__(args)

        self.num_lines = args.num_lines
        self.img_size = args.img_size

        self.genotype_size = ((3 * 6) + 7)  # 3 segments, 6 values each, plus color (4), width (1), starting point (2)
        self.real_genotype_size = self.genotype_size * self.num_lines

    def chunks(self, array):
        img = np.array(array)
        return np.reshape(img, (self.args.num_lines, self.genotype_size))

    def __str__(self):
        return "clipdrawer"

    def render(self, a, cur_iteration):
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

        ims = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.img_size, self.img_size)
        cr = cairo.Context(ims)

        cr.set_source_rgba(R, G, B, 1.0)  # everythingon cairo appears to be between 0 and 1
        cr.rectangle(0, 0, self.img_size, self.img_size)  # define a recatangle and then fill it
        cr.fill()

        cr.set_line_cap(cairo.LINE_CAP_ROUND)
        cr.set_line_join(cairo.LINE_JOIN_ROUND)

        max_width = 2.0 * self.img_size / 100
        min_width = 0.5 * self.img_size / 100

        for e in rest:
            R = e[0]
            G = e[1]
            B = e[2]
            A = e[3]

            w = map_number(e[4], 0, 1, min_width, max_width)

            cr.set_source_rgba(R, G, B, A)
            # line width
            cr.set_line_width(w)

            num_segments = 3
            p0 = (e[5], e[6])

            ind = 7
            for j in range(num_segments):
                radius = 0.1
                p1 = (p0[0] + radius * (e[ind] - 0.5), p0[1] + radius * (e[ind + 1] - 0.5))
                p2 = (p1[0] + radius * (e[ind + 2] - 0.5), p1[1] + radius * (e[ind + 3] - 0.5))
                p3 = (p2[0] + radius * (e[ind + 4] - 0.5), p2[1] + radius * (e[ind + 5] - 0.5))

                ind += 6

                cr.move_to(p0[0] * self.img_size, p0[1] * self.img_size)
                cr.curve_to(p1[0] * self.img_size, p1[1] * self.img_size, p2[0] * self.img_size, p2[1] * self.img_size, p3[0] * self.img_size,
                            p3[1] * self.img_size)

                cr.stroke()

                p0 = p3

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
