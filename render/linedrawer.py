import cairo
import numpy as np
from PIL import Image

from renderinterface import RenderingInterface
from utils import map_number


class LineRenderer(RenderingInterface):
    def __init__(self, args):
        super(LineRenderer, self).__init__(args)

        self.num_lines = args.num_lines

        self.genotype_size = 12
        self.real_genotype_size = self.genotype_size * args.num_lines

    def chunks(self, array):
        img = np.array(array)
        return np.reshape(img, (self.args.num_lines, self.genotype_size))

    def __str__(self):
        return "clipdrawer"

    def render(self, a, img_size):
        ims = cairo.ImageSurface(cairo.FORMAT_ARGB32, img_size, img_size)
        cr = cairo.Context(ims)

        max_width = 2 * img_size / 100
        min_width = 0.5 * img_size / 100

        # background shape
        p0 = [0, 0]
        p1 = [img_size, img_size]
        cr.set_source_rgba(242/255.0, 238/255.0, 203/255.0, 1.0)  # everything on cairo appears to be between 0 and 1
        cr.rectangle(0, 0, img_size, img_size)  # define a rectangle and then fill it
        cr.fill()

        """
        # Initialize Random Curves
        for i in range(num_paths):
            num_segments = self.stroke_length
            num_control_points = torch.zeros(num_segments, dtype = torch.int32) + 2
            points = []
            radius = 0.5
            radius_x = 0.5 #radius * canvas_height / canvas_width
            p0 = (0.5 + radius_x * (random.random() - 0.5), 0.5 + radius * (random.random() - 0.5))
            points.append(p0)
            for j in range(num_segments):
                radius = 1.0 / (num_segments + 2)
                radius_x = radius * canvas_height / canvas_width
                p1 = (p0[0] + radius_x * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                p2 = (p1[0] + radius_x * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
                p3 = (p2[0] + radius_x * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
                points.append(p1)
                points.append(p2)
                points.append(p3)
                p0 = (bound(p3[0],0,1), bound(p3[1],0,1))

            points = torch.tensor(points)
            points[:, 0] *= canvas_width
            points[:, 1] *= canvas_height
            path = pydiffvg.Path(num_control_points = num_control_points, points = points, stroke_width = torch.tensor(max_width/10), is_closed = False)
            shapes.append(path)
            s_col = [0, 0, 0, 1]
            path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes)-1]), fill_color = None, stroke_color = torch.tensor(s_col))
            shape_groups.append(path_group)

        # Just some diffvg setup
        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            canvas_width, canvas_height, shapes, shape_groups)
        render = pydiffvg.RenderFunction.apply
        img = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)
        """