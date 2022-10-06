import math

import cairo
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF

from render.renderinterface import RenderingInterface
from utils import map_number


class CharsRenderer(RenderingInterface):
    def __init__(self, args):
        super(CharsRenderer, self).__init__(args)

        self.img_size = args.img_size
        self.num_lines = args.num_lines

        self.device = args.device

        self.header_length = 1

        self.genotype_size = 8

    def generate_individual(self):
        return np.random.rand(self.num_lines, self.genotype_size).flatten()

    def chunks(self, array):
        return np.reshape(array, (self.num_lines, self.genotype_size))

    def to_adam(self, individual):
        ind_copy = np.copy(individual)
        ind_copy = self.chunks(ind_copy)
        ind_copy = torch.tensor(ind_copy).float().to(self.device)
        ind_copy.requires_grad = True
        return [ind_copy]

    def get_individual(self, adam_ind):
        return adam_ind[0].cpu().detach().numpy().flatten()

    def __str__(self):
        return "chars"

    def render(self, input_ind):
        """
        @x: array of real vectors, length 8, each component normalized 0-1
        """
        input_ind = input_ind[0]
        input_ind = input_ind.cpu().detach().numpy()

        chars = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
                 "u", "w", "y", "x", "z"]
        # split input array into header and rest
        head = input_ind[:self.header_length]
        rest = input_ind[self.header_length:]

        # determine background color from header
        r = head[0][0]
        g = head[0][1]
        b = head[0][2]

        # create the image and drawing context
        # im = Image.new('RGB', (size, size), (R, G, B))
        # draw = ImageDraw.Draw(im, 'RGB')

        ims = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.img_size, self.img_size)
        cr = cairo.Context(ims)

        cr.set_source_rgba(r, g, b, 1.0)  # everythingon cairo appears to be between 0 and 1
        cr.rectangle(0, 0, self.img_size, self.img_size)  # define a rectangle and then fill it
        cr.fill()

        # now draw lines
        min_size = 0.01 * self.img_size
        max_size = self.img_size
        cr.set_line_cap(cairo.LINE_CAP_ROUND)
        cr.set_line_join(cairo.LINE_JOIN_ROUND)
        cr.select_font_face("Serif", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        charit = 0

        for e in rest:
            # determine foreground color from header
            r = e[0]
            g = e[1]
            b = e[2]

            w = map_number(e[4], 0, 1, min_size, max_size)

            # tamanho vem do 4
            cr.set_font_size(w)

            # cr.set_source_rgb(R, G, B)
            # line width
            # cr.set_line_width(w)

            # cr.set_line_width(1)

            cr.set_source_rgb(r, g, b)

            # cr.set_font_size(100)

            # aqui devia ser o extent, para definirmos
            (xc, yc, widthc, heightc, dx, dy) = cr.text_extents(chars[charit])

            # x = map_number(e[5], 0, 1, 0-widthc, size)
            # y = map_number(e[6], 0, 1, 0-heightc-yc, size-yc)
            angle = map_number(e[7], 0, 1, 0, math.pi * 2)

            input_ind = map_number(e[5], 0, 1, 0 - widthc - xc, self.img_size - xc)
            y = map_number(e[6], 0, 1, 0 - heightc - yc, self.img_size - yc)

            # cr.move_to(x, y)
            # cr.show_text(chars[charit])

            # cr.save()
            # cr.move_to(x, y+900)
            #  cr.rotate(angle)
            #  cr.show_text(chars[charit])
            #  cr.restore()

            cr.save()

            nx = -widthc / 2.0
            ny = heightc / 2

            cr.translate(input_ind - nx, y - ny)
            cr.rotate(angle)
            cr.translate(nx, ny)
            cr.move_to(0, 0)
            cr.show_text(chars[charit])
            cr.restore()

            charit = (charit + 1) % len(chars)

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
        return TF.to_tensor(pilImage).unsqueeze(0)
