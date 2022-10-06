import cairo
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

from render.renderinterface import RenderingInterface
from utils import map_number


class PylinhasRenderer(RenderingInterface):
    def __init__(self, args):
        super(PylinhasRenderer, self).__init__(args)

        self.img_size = args.img_size
        self.num_lines = args.num_lines

        self.device = args.device

        self.header_length = 1

        self.genotype_size = 8

    def chunks(self, array):
        return np.reshape(array, (self.num_lines, self.genotype_size))

    def generate_individual(self):
        return np.random.rand(self.num_lines, self.genotype_size).flatten()

    def to_adam(self, individual):
        ind_copy = np.copy(individual)
        ind_copy = self.chunks(ind_copy)
        ind_copy = torch.tensor(ind_copy).float().to(self.device)
        ind_copy.requires_grad = True
        return [ind_copy]

    def get_individual(self, adam_ind):
        return adam_ind[0].cpu().detach().numpy().flatten()

    def __str__(self):
        return "pylinhas"

    def render(self, input_ind):
        input_ind = input_ind[0]
        input_ind = input_ind.cpu().detach().numpy()

        # split input array into header and rest
        head = input_ind[:self.header_length]
        rest = input_ind[self.header_length:]

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

        # now draw lines
        if len(head[0]) > 8:
            min_width = 0.004 * self.img_size
            max_width = 0.04 * self.img_size
        else:
            min_width = 0.0001 * self.img_size
            max_width = 0.1 * self.img_size

        cr.set_line_cap(cairo.LINE_CAP_ROUND)
        cr.set_line_join(cairo.LINE_JOIN_ROUND)

        for e in rest:
            # determine foreground color from header
            R = e[0]
            G = e[1]
            B = e[2]
            w = map_number(e[3], 0, 1, min_width, max_width)

            cr.set_source_rgb(R, G, B)
            # line width
            cr.set_line_width(w)

            # cr.set_line_width(1)

            for it in range(4, len(e) - 1, 2):
                input_ind = map_number(e[it], 0, 1, 0, self.img_size)
                y = map_number(e[it + 1], 0, 1, 0, self.img_size)
                cr.line_to(input_ind, y)

            cr.close_path()
            cr.stroke_preserve()
            cr.fill()

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
        return TF.to_tensor(pilImage).unsqueeze(0)
