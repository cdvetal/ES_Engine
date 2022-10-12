import random

import numpy as np
import pydiffvg
import torch

from render.renderinterface import RenderingInterface


class LineDrawRenderer(RenderingInterface):
    def __init__(self, args):
        super(LineDrawRenderer, self).__init__(args)

        self.device = args.device

        self.num_lines = args.num_lines
        self.img_size = args.img_size

        self.max_width = 2 * self.img_size / 100
        self.min_width = 0.5 * self.img_size / 100

        self.stroke_length = 8

    def chunks(self, array):
        # array = torch.tensor(array, dtype=torch.float)
        # return array.view(self.num_lines, (self.num_segments * 3) + 1, 2)
        return np.reshape(array, (self.num_lines, (self.stroke_length * 3) + 1, 2))

    def bound(self, value, low, high):
        return max(low, min(high, value))

    def generate_individual(self):
        # Initialize Random Curves
        individual = []

        # Initialize Random Curves
        for i in range(self.num_lines):
            num_segments = self.stroke_length
            # num_control_points = torch.zeros(num_segments, dtype=torch.int32) + 2
            points = []
            radius = 0.5
            p0 = (0.5 + radius * (random.random() - 0.5), 0.5 + radius * (random.random() - 0.5))
            points.append(p0)
            for j in range(num_segments):
                radius = 1.0 / (num_segments + 2)
                p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
                p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
                points.append(p1)
                points.append(p2)
                points.append(p3)
                p0 = (self.bound(p3[0], 0, 1), self.bound(p3[1], 0, 1))
            points = torch.tensor(points)
            # points[:, 0] *= self.img_size
            # points[:, 1] *= self.img_size
            individual.append(np.array(points))

        individual = np.array(individual)
        return individual.flatten()

    def get_individual(self, _):
        individual = []
        for path in self.shapes[1:]:
            points = path.points.clone().detach()
            points /= self.img_size
            individual.append(points.cpu().detach().numpy())

        individual = np.array(individual).flatten()
        return individual

    def to_adam(self, individual):
        ind_copy = np.copy(individual)

        ind_copy = self.chunks(ind_copy)
        ind_copy = torch.tensor(ind_copy).float().to(self.device)

        shapes = []
        shape_groups = []

        # background shape
        p0 = [0, 0]
        p1 = [self.img_size, self.img_size]
        path = pydiffvg.Rect(p_min=torch.tensor(p0), p_max=torch.tensor(p1))
        shapes.append(path)
        # https://encycolorpedia.com/f2eecb
        cell_color = torch.tensor([242 / 255.0, 238 / 255.0, 203 / 255.0, 1.0])
        path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]), stroke_color=None,
                                         fill_color=cell_color)
        shape_groups.append(path_group)

        # Initialize Random Curves
        for i in range(self.num_lines):
            num_segments = self.stroke_length
            num_control_points = torch.zeros(num_segments, dtype=torch.int32) + 2
            points = []
            p0 = (ind_copy[i][0][0], ind_copy[i][0][1])
            points.append(p0)
            for j in range(num_segments):
                p1 = (ind_copy[i][(j * 3) + 1][0], ind_copy[i][(j * 3) + 1][1])
                p2 = (ind_copy[i][(j * 3) + 2][0], ind_copy[i][(j * 3) + 2][1])
                p3 = (ind_copy[i][(j * 3) + 3][0], ind_copy[i][(j * 3) + 3][1])
                points.append(p1)
                points.append(p2)
                points.append(p3)
            points = torch.tensor(points)
            points[:, 0] *= self.img_size
            points[:, 1] *= self.img_size
            path = pydiffvg.Path(num_control_points=num_control_points, points=points,
                                 stroke_width=torch.tensor(self.max_width / 10), is_closed=False)
            shapes.append(path)
            s_col = [0, 0, 0, 1]
            path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]), fill_color=None,
                                             stroke_color=torch.tensor(s_col))
            shape_groups.append(path_group)

        points_vars = []
        for path in shapes[1:]:
            path.points.requires_grad = True
            points_vars.append(path.points)

        self.shapes = shapes
        self.shape_groups = shape_groups

        return points_vars

    def __str__(self):
        return "linedraw"

    def render(self, input_ind):
        render = pydiffvg.RenderFunction.apply
        scene_args = pydiffvg.RenderFunction.serialize_scene(self.img_size, self.img_size, self.shapes, self.shape_groups)
        img = render(self.img_size, self.img_size, 2, 2, 0, None, *scene_args)
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3,
                                                          device=pydiffvg.get_device()) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2)  # NHWC -> NCHW

        pydiffvg.save_svg('out.svg', self.img_size, self.img_size, self.shapes, self.shape_groups)
        return img
