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
    if (len(head[0]) > 8):
        min_width = 0.004 * size
        max_width = 0.04 * size
    else:
        min_width = 0.0001 * size
        max_width = 0.1 * size

    cr.set_line_cap(cairo.LINE_CAP_ROUND)
    cr.set_line_join(cairo.LINE_JOIN_ROUND)

    for e in rest:
        # determine foreground color from header
        R = e[0]
        G = e[1]
        B = e[2]
        w = map_number(e[4], 0, 1, min_width, max_width)

        cr.set_source_rgb(R, G, B)
        # line width
        cr.set_line_width(w)

        # cr.set_line_width(1)

        for it in range(4, len(e)-1, 2):
            x = map_number(e[it], 0, 1, 0, size)
            y = map_number(e[it+1], 0, 1, 0, size)
            cr.line_to(x, y)

        cr.close_path()
        cr.stroke_preserve()
        cr.fill()

    pilMode = 'RGB'
    #argbArray = numpy.fromstring( ims.get_data(), 'c' ).reshape( -1, 4 )
    argbArray = np.fromstring(bytes(ims.get_data()), 'c').reshape(-1, 4)
    rgbArray = argbArray[:, 2::-1]
    pilData = rgbArray.reshape(-1).tostring()
    pilImage = Image.frombuffer(pilMode,
                                (ims.get_width(), ims.get_height()), pilData, "raw",
                                pilMode, 0, 1)
    pilImage = pilImage.convert('RGB')
    #pilImage.show() # mostra a image no preview

    # draw line with round line caps (circles at the end)
    # draw.polygon(
    #   ((x1,y1), (x2,y2), (x3,y3)), (R,G,B), outline=None)
    return pilImage
