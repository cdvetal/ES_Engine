import sys
import math
import cairo

from PIL import Image, ImageDraw
#from gi.repository import Gtk, Gdk

import numpy as np
import csv

# canonical interpolation function, like https://p5js.org/reference/#/p5/map


def map_number(n, start1, stop1, start2, stop2):
    return ((n-start1)/(stop1-start1))*(stop2-start2)+start2

# input: array of real vectors, length 8, each component normalized 0-1


def render(a, size):

    # split input array into header and rest
    header_length = 1
    head = a[:header_length]
    rest = a[header_length:]

    # determine background color from header
    R = head[0][0]
    G = head[0][1]
    B = head[0][2]

    # create the image and drawing context
    #im = Image.new('RGB', (size, size), (R, G, B))
    #draw = ImageDraw.Draw(im, 'RGB')

    ims = cairo.ImageSurface(cairo.FORMAT_ARGB32, size, size)
    cr = cairo.Context(ims)

    cr.set_source_rgba(R, G, B, 1.0)  # everythingon cairo appears to be between 0 and 1
    cr.rectangle(0, 0, size, size)  # define a recatangle and then fill it
    cr.fill()

    # now draw lines

    min_size = 0.005 * size
    max_size = 0.15 * size

    cr.set_line_cap(cairo.LINE_CAP_ROUND)
    cr.set_line_join(cairo.LINE_JOIN_ROUND)

    # main cicle begins

    for e in rest:
        R = e[0]
        G = e[1]
        B = e[2]
        w1 = map_number(e[3], 0, 1, min_size, max_size)
        w2 = map_number(e[4], 0, 1, min_size, max_size)

        a = Vector(map_number(e[5], 0, 1, 0, size), map_number(e[6], 0, 1, 0, size))
        # a=Vector(500, 200)
        d = Vector(map_number(e[7], 0, 1, 0, size), map_number(e[8], 0, 1, 0, size))
        #d=Vector(200, 800)
        A = Vector(a.x, a.y)
        D = Vector(d.x, d.y)

        AD = D-A
        maxdist = AD.length()/2

        b = a+Vector(map_number(e[9], 0, 1, 0, maxdist) * np.sign(d.x-a.x), map_number(e[10], 0, 1, 0, maxdist) * np.sign(d.y-a.y))

        # b=a+Vector(-100, +150) #os sinais têm que ser compativeis com as direções
        # c=d+Vector(-100, -100) #os sinais têm que ser compativeis com as direções
        c = d+Vector(map_number(e[11], 0, 1, 0, maxdist) * np.sign(a.x-d.x), map_number(e[12], 0, 1, 0, maxdist) * np.sign(a.y-d.y))

        cr.set_source_rgb(R, G, B)
        cr.set_line_width(1)

        cr.move_to(a.x, a.y)
        cr.curve_to(b.x, b.y, c.x, c.y, d.x, d.y)
        cr.stroke()

        B = Vector(b.x, b.y)
        AB = B - A
        AB_perp_normed = perpendicular(normalize(AB))
        a1 = A + AB_perp_normed * w1
        a2 = A - AB_perp_normed * w1
        b1 = B + AB_perp_normed * w1
        b2 = B - AB_perp_normed * w1

        C = Vector(c.x, c.y)
        CD = D-C
        CD_perp_normed = perpendicular(normalize(CD))
        c1 = C + CD_perp_normed * w2
        c2 = C - CD_perp_normed * w2
        d1 = D + CD_perp_normed * w2
        d2 = D - CD_perp_normed * w2

        cr.move_to(a2.x, a2.y)
        cr.line_to(a1.x, a1.y)
        cr.curve_to(b1.x, b1.y, c1.x, c1.y, d1.x, d1.y)
        cr.line_to(d2.x, d2.y)

        cr.move_to(d2.x, d2.y)
        cr.curve_to(c2.x, c2.y, b2.x, b2.y, a2.x, a2.y)
        cr.fill_preserve()
        cr.stroke()

        cr.arc(a.x, a.y, w1, AB_perp_normed.angle(), math.pi+AB_perp_normed.angle())
        cr.fill_preserve()
        cr.stroke()

        cr.arc(d.x, d.y, w2, math.pi+CD_perp_normed.angle(), CD_perp_normed.angle())
        cr.fill_preserve()
        cr.stroke()

    pilMode = 'RGB'
    #argbArray = numpy.fromstring( ims.get_data(), 'c' ).reshape( -1, 4 )
    argbArray = np.fromstring(bytes(ims.get_data()), 'c').reshape(-1, 4)
    rgbArray = argbArray[:, 2::-1]
    pilData = rgbArray.reshape(-1).tostring()
    pilImage = Image.frombuffer(pilMode,
                                (ims.get_width(), ims.get_height()), pilData, "raw",
                                pilMode, 0, 1)
    pilImage = pilImage.convert('RGB')

    return pilImage


def normalize(coord):
    return Vector(
        coord.x/coord.length(),
        coord.y/coord.length()
    )


def perpendicular(coord):
    # Shifts the angle by pi/2 and calculate the coordinates
    # using the original vector length
    return Vector(
        coord.length()*math.cos(coord.angle()+math.pi/2),
        coord.length()*math.sin(coord.angle()+math.pi/2)
    )


class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    def length(self):
        # Returns the length of the vector
        return math.sqrt(self.x**2 + self.y**2)

    def angle(self):
        # Returns the vector's angle
        return math.atan2(self.y, self.x)

    def norm(self):
        return self.dot(self)**0.5

    def normalized(self):
        norm = self.norm()
        return Vector(self.x / norm, self.y / norm)

    def perp(self):
        return Vector(1, -self.x / self.y)

    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)

    def __str__(self):
        return f'({self.x}, {self.y})'
