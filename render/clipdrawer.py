import random

import numpy as np
import torch
import pydiffvg

from render.renderinterface import RenderingInterface


class ClipDrawRenderer(RenderingInterface):
    def __init__(self, args):
        super(ClipDrawRenderer, self).__init__(args)

        self.device = args.device

        self.lr = 1.0

        self.num_lines = args.num_lines
        self.img_size = args.img_size

        self.num_segments = 2
        self.max_width = 5 * self.img_size / 100
        self.min_width = 1 * self.img_size / 100

        self.x = None

    def chunks(self, array):
        return np.reshape(array, (self.num_lines, (self.num_segments * 3) + 1, 2))

    def generate_individual(self):
        # Initialize Random Curves
        individual = []
        for i in range(self.num_lines):
            points = []
            p0 = (random.random(), random.random())
            points.append(p0)
            for j in range(self.num_segments):
                radius = 0.1
                p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
                p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
                points.append(p1)
                points.append(p2)
                points.append(p3)
                p0 = p3
            points = torch.tensor(points)
            # points[:, 0] *= self.img_size
            # points[:, 1] *= self.img_size
            individual.append(np.array(points))

        individual = np.array(individual)
        print(individual.shape)

        return individual.flatten()

    def get_individual(self):
        individual = []
        for path in self.shapes:
            points = torch.tensor(path.points)
            points /= self.img_size
            individual.append(points.cpu().detach().numpy())

        individual = np.array(individual).flatten()
        # print(individual.shape)
        return individual

    def to_adam(self, individual, gradients=True):
        ind_copy = np.copy(individual)

        ind_copy = self.chunks(ind_copy)
        ind_copy = torch.tensor(ind_copy).float().to(self.device)

        # Initialize Random Curves
        shapes = []
        shape_groups = []
        for i in range(self.num_lines):
            # num_segments = random.randint(1, 3)
            num_control_points = torch.zeros(self.num_segments, dtype=torch.int32) + 2
            points = []
            p0 = (ind_copy[i][0][0], ind_copy[i][0][1])
            points.append(p0)
            for j in range(self.num_segments):
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
                                 stroke_width=torch.tensor((self.min_width + self.max_width) / 4), is_closed=False)
            shapes.append(path)
            path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]), fill_color=None,
                                             stroke_color=torch.tensor(
                                                 [random.random(), random.random(), random.random(), 1.0]))
            shape_groups.append(path_group)

        points_vars = []
        stroke_width_vars = []
        color_vars = []
        for path in shapes:
            if gradients:
                path.points.requires_grad = True
            points_vars.append(path.points)

            if gradients:
                path.stroke_width.requires_grad = True
            stroke_width_vars.append(path.stroke_width)
        for group in shape_groups:
            if gradients:
                group.stroke_color.requires_grad = True
            color_vars.append(group.stroke_color)

        points_optim = torch.optim.Adam(points_vars, lr=1.0)
        width_optim = torch.optim.Adam(stroke_width_vars, lr=0.1)
        color_optim = torch.optim.Adam(color_vars, lr=0.01)

        self.shapes = shapes
        self.shape_groups = shape_groups

        return [points_optim, width_optim, color_optim]

    def __str__(self):
        return "clipdraw"

    def render(self):
        render = pydiffvg.RenderFunction.apply

        scene_args = pydiffvg.RenderFunction.serialize_scene(
            self.img_size, self.img_size, self.shapes, self.shape_groups)
        img = render(self.img_size, self.img_size, 2, 2, 0, None, *scene_args)
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3,
                                                          device=pydiffvg.get_device()) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2)  # NHWC -> NCHW
        return img
