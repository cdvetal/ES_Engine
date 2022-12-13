import random

import numpy as np
import pydiffvg
import torch

from render.renderinterface import RenderingInterface


def rect_from_corners(p0, p1):
    x1, y1 = p0
    x2, y2 = p1
    pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    return pts


# canonical interpolation function, like https://p5js.org/reference/#/p5/map
def map_number(n, start1, stop1, start2, stop2):
    return ((n - start1) / (stop1 - start1)) * (stop2 - start2) + start2


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


class PixelRenderer(RenderingInterface):
    def __init__(self, args):
        super(PixelRenderer, self).__init__(args)

        self.device = args.device

        self.num_lines = args.num_lines
        self.img_size = args.img_size

        self.lr = 0.03

        self.pixel_type = "rect"

        self.cell_size = self.img_size / self.num_lines

    def chunks(self, array):
        return np.reshape(array, (self.num_lines, self.num_lines, 4))

    def __str__(self):
        return "pixeldraw"

    def generate_individual(self):
        # Initialize Random Pixels
        colors = []
        for r in range(self.num_lines):
            num_cols_this_row = self.num_lines
            for c in range(num_cols_this_row):
                cell_color = torch.tensor([random.random(), random.random(), random.random(), 1.0])
                colors.append(np.array(cell_color))

        individual = np.array(colors)
        return individual.flatten()

    def get_individual(self):
        individual = []
        for group in self.shape_groups:
            colors = group.fill_color.clone().detach()
            individual.append(colors.cpu().detach().numpy())

        individual = np.array(individual).flatten()
        return individual

    def to_adam(self, individual, gradients=True):
        ind_copy = np.copy(individual)

        ind_copy = self.chunks(ind_copy)
        ind_copy = torch.tensor(ind_copy).float().to(self.device)

        # Initialize Random Pixels
        shapes = []
        shape_groups = []
        colors = []
        for r in range(self.num_lines):
            cur_y = r * self.cell_size
            num_cols_this_row = self.num_lines
            for c in range(num_cols_this_row):
                cur_x = c * self.cell_size
                cell_color = torch.tensor([ind_copy[r][c][0], ind_copy[r][c][1], ind_copy[r][c][2], 1.0])
                colors.append(cell_color)
                p0 = [cur_x, cur_y]
                p1 = [cur_x + self.cell_size, cur_y + self.cell_size]

                if self.pixel_type == "tri":
                    pts = tri_from_corners(p0, p1, (r + c) % 2 == 0)
                elif self.pixel_type == "knit":
                    pts = knit_from_corners(p0, p1)
                else:
                    pts = rect_from_corners(p0, p1)
                pts = torch.tensor(pts, dtype=torch.float32).view(-1, 2)
                # path = pydiffvg.Polygon(pts, True)

                path = pydiffvg.Rect(p_min=torch.tensor(p0), p_max=torch.tensor(p1))
                shapes.append(path)
                path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]), stroke_color=None,
                                                 fill_color=cell_color)
                shape_groups.append(path_group)

        color_vars = []
        for group in shape_groups:
            if gradients:
                group.fill_color.requires_grad = True
            color_vars.append(group.fill_color)

        color_optim = torch.optim.Adam(color_vars, lr=0.01)

        self.shapes = shapes
        self.shape_groups = shape_groups

        return [color_optim]

    def render(self):
        render = pydiffvg.RenderFunction.apply
        scene_args = pydiffvg.RenderFunction.serialize_scene(self.img_size, self.img_size, self.shapes,
                                                             self.shape_groups)
        img = render(self.img_size, self.img_size, 2, 2, 0, None, *scene_args)
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3,
                                                          device=pydiffvg.get_device()) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2)  # NHWC -> NCHW

        return img
