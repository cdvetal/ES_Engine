import cairo
import numpy as np
from PIL import Image

from render.renderinterface import RenderingInterface


def rect_from_corners(p0, p1):
    x1, y1 = p0
    x2, y2 = p1
    pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    return pts


# canonical interpolation function, like https://p5js.org/reference/#/p5/map
def map_number(n, start1, stop1, start2, stop2):
    return ((n - start1) / (stop1 - start1)) * (stop2 - start2) + start2


def diamond_from_corners(p0, p1):
    x1, y1 = p0
    x2, y2 = p1
    n = 1
    hyA = map_number(-2, -n, n, y1, y2)
    hyB = map_number(2, -n, n, y1, y2)
    hyH = map_number(0, -n, n, y1, y2)
    hxH = map_number(0, -n, n, x1, x2)
    pts = [[hxH, hyA], [x1, hyH], [hxH, hyB], [x2, hyH]]
    return pts


def tri_from_corners(p0, p1, is_up):
    x1, y1 = p0
    x2, y2 = p1
    n = 1
    hxA = map_number(2, -n, n, x1, x2)
    hxB = map_number(-2, -n, n, x1, x2)
    hxH = map_number(0, -n, n, x1, x2)
    if is_up:
        pts = [[hxH, y1], [hxB, y2], [hxA, y2]]
    else:
        pts = [[hxH, y2], [hxA, y1], [hxB, y1]]
    return pts


def hex_from_corners(p0, p1):
    x1, y1 = p0
    x2, y2 = p1
    n = 3
    hyA = map_number(4, -n, n, y1, y2)
    hyB = map_number(2, -n, n, y1, y2)
    hyC = map_number(-2, -n, n, y1, y2)
    hyD = map_number(-4, -n, n, y1, y2)
    hxH = map_number(0, -n, n, x1, x2)
    pts = [[hxH, hyA], [x1, hyB], [x1, hyC], [hxH, hyD], [x2, hyC], [x2, hyB]]
    return pts


def knit_from_corners(p0, p1):
    x1, y1 = p0
    x2, y2 = p1
    xm = (x1 + x2) / 2.0
    lean_up = 0.45
    slump_down = 0.30
    fall_back = 0.2
    y_up1 = map_number(lean_up, 0, 1, y2, y1)
    y_up2 = map_number(1 + lean_up, 0, 1, y2, y1)
    y_down1 = map_number(slump_down, 0, 1, y1, y2)
    y_down2 = map_number(1 + slump_down, 0, 1, y1, y2)
    x_fall_back1 = map_number(fall_back, 0, 1, x2, xm)
    x_fall_back2 = map_number(fall_back, 0, 1, x1, xm)

    pts = []
    # center bottom
    pts.append([xm, y_down2])
    # vertical line on right side
    pts.extend([[x2, y_up1], [x2, y_up2]])
    # horizontal line back
    pts.append([x_fall_back1, y_up2])
    # center top
    pts.append([xm, y_down1])
    # back up to top
    pts.append([x_fall_back2, y_up2])
    # vertical line on left side
    pts.extend([[x1, y_up2], [x1, y_up1]])
    return pts


# canonical interpolation function, like https://p5js.org/reference/#/p5/map
def map_number(n, start1, stop1, start2, stop2):
    return ((n - start1) / (stop1 - start1)) * (stop2 - start2) + start2;


def diamond_from_corners(p0, p1):
    x1, y1 = p0
    x2, y2 = p1
    n = 1
    hyA = map_number(-2, -n, n, y1, y2)
    hyB = map_number(2, -n, n, y1, y2)
    hyH = map_number(0, -n, n, y1, y2)
    hxH = map_number(0, -n, n, x1, x2)
    pts = [[hxH, hyA], [x1, hyH], [hxH, hyB], [x2, hyH]]
    return pts


def tri_from_corners(p0, p1, is_up):
    x1, y1 = p0
    x2, y2 = p1
    n = 1
    hxA = map_number(2, -n, n, x1, x2)
    hxB = map_number(-2, -n, n, x1, x2)
    hxH = map_number(0, -n, n, x1, x2)
    if is_up:
        pts = [[hxH, y1], [hxB, y2], [hxA, y2]]
    else:
        pts = [[hxH, y2], [hxA, y1], [hxB, y1]]
    return pts


def hex_from_corners(p0, p1):
    x1, y1 = p0
    x2, y2 = p1
    n = 3
    hyA = map_number(4, -n, n, y1, y2)
    hyB = map_number(2, -n, n, y1, y2)
    hyC = map_number(-2, -n, n, y1, y2)
    hyD = map_number(-4, -n, n, y1, y2)
    hxH = map_number(0, -n, n, x1, x2)
    pts = [[hxH, hyA], [x1, hyB], [x1, hyC], [hxH, hyD], [x2, hyC], [x2, hyB]]
    return pts


def knit_from_corners(p0, p1):
    x1, y1 = p0
    x2, y2 = p1
    xm = (x1 + x2) / 2.0
    lean_up = 0.45
    slump_down = 0.30
    fall_back = 0.2
    y_up1 = map_number(lean_up, 0, 1, y2, y1)
    y_up2 = map_number(1 + lean_up, 0, 1, y2, y1)
    y_down1 = map_number(slump_down, 0, 1, y1, y2)
    y_down2 = map_number(1 + slump_down, 0, 1, y1, y2)
    x_fall_back1 = map_number(fall_back, 0, 1, x2, xm)
    x_fall_back2 = map_number(fall_back, 0, 1, x1, xm)

    pts = []
    # center bottom
    pts.append([xm, y_down2])
    # vertical line on right side
    pts.extend([[x2, y_up1], [x2, y_up2]])
    # horizontal line back
    pts.append([x_fall_back1, y_up2])
    # center top
    pts.append([xm, y_down1])
    # back up to top
    pts.append([x_fall_back2, y_up2])
    # vertical line on left side
    pts.extend([[x1, y_up2], [x1, y_up1]])
    return pts


shift_pixel_types = ["hex", "rectshift", "diamond"]


class PixelRenderer(RenderingInterface):
    def __init__(self, args):
        super(PixelRenderer, self).__init__(args)

        self.num_lines = args.num_lines
        self.img_size = args.img_size

        self.pixel_type = "rect"

        self.canvas_size = self.img_size
        self.cell_size = self.canvas_size / self.num_lines

        self.genotype_size = 3
        self.real_genotype_size = self.genotype_size * (args.num_lines * args.num_lines)

    def chunks(self, array):
        img = np.array(array)
        return np.reshape(img, (self.num_lines, self.num_lines, self.genotype_size))

    def __str__(self):
        return "pixel"

    def render(self, a, cur_iteration):
        ims = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.img_size, self.img_size)
        cr = cairo.Context(ims)

        cr.set_source_rgba(1.0, 1.0, 1.0, 1.0)  # everything on cairo appears to be between 0 and 1
        cr.rectangle(0, 0, self.img_size, self.img_size)  # define a rectangle and then fill it
        cr.fill()

        for r in range(self.num_lines):
            cur_y = int(r * self.cell_size)
            for c in range(self.num_lines):
                cur_x = c * self.cell_size

                p0 = [cur_x, cur_y]
                p1 = [cur_x + self.cell_size, cur_y + self.cell_size]

                if self.pixel_type == "hex":
                    pts = hex_from_corners(p0, p1)
                elif self.pixel_type == "tri":
                    pts = tri_from_corners(p0, p1, (r + c) % 2 == 0)
                elif self.pixel_type == "diamond":
                    pts = diamond_from_corners(p0, p1)
                elif self.pixel_type == "knit":
                    pts = knit_from_corners(p0, p1)
                else:
                    pts = rect_from_corners(p0, p1)

                pts = np.array(pts)
                # print("Shape", pts.shape)

                red = a[r][c][0]
                green = a[r][c][1]
                blue = a[r][c][2]

                cr.set_source_rgb(red, green, blue)

                cr.move_to(pts[0][0], pts[0][1])

                for p in range(1, len(pts)):
                    cr.line_to(pts[p][0], pts[p][1])

                cr.close_path()

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
        return pilImage