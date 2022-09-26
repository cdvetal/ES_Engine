import random

import numpy as np
import torch
from PIL import Image

from render.renderinterface import RenderingInterface
from utils import map_number


class ClipDrawRenderer(RenderingInterface):
    def __init__(self, args):
        super(ClipDrawRenderer, self).__init__(args)

        self.num_lines = args.num_lines
        self.img_size = args.img_size

    def chunks(self, array):
        array = torch.tensor(array, dtype=torch.float)
        return array.view(self.num_lines, self.genotype_size)

    def generate_individual(self):
        # Use GPU if available
        pydiffvg.set_use_gpu(torch.cuda.is_available())
        device = torch.device('cuda')
        pydiffvg.set_device(device)

        canvas_width, canvas_height = self.img_size, self.img_size
        num_paths = self.num_lines
        max_width = 5 * canvas_height / 100
        min_width = 1 * canvas_height / 100

        # Initialize Random Curves
        shapes = []
        shape_groups = []
        for i in range(num_paths):
            num_segments = random.randint(1, 3)
            num_control_points = torch.zeros(num_segments, dtype=torch.int32) + 2
            points = []
            p0 = (random.random(), random.random())
            points.append(p0)
            for j in range(num_segments):
                radius = 0.1
                p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
                p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
                points.append(p1)
                points.append(p2)
                points.append(p3)
                p0 = p3
            points = torch.tensor(points)
            points[:, 0] *= canvas_width
            points[:, 1] *= canvas_height
            path = pydiffvg.Path(num_control_points=num_control_points, points=points,
                                 stroke_width=torch.tensor((min_width + max_width) / 4), is_closed=False)
            shapes.append(path)
            path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]), fill_color=None,
                                             stroke_color=torch.tensor(
                                                 [random.random(), random.random(), random.random(), random.random()]))
            shape_groups.append(path_group)

        points_vars = []
        for path in shapes:
            path.points.requires_grad = True
            points_vars.append(path.points)

        # self.points_vars = points_vars
        self.shapes = shapes
        self.shape_groups = shape_groups

        return points_vars

    def __str__(self):
        return "clipdrawer"

    def render(self, a, cur_iteration):
        render = pydiffvg.RenderFunction.apply
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            self.img_size, self.img_size, self.shapes, self.shape_groups)
        img = render(self.img_size, self.img_size, 2, 2, cur_iteration, None, *scene_args)
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3,
                                                          device=pydiffvg.get_device()) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2)  # NHWC -> NCHW
        return img
