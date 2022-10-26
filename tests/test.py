
import numpy as np
import torch

n_points = 5
img_size = 256
individual = np.random.randn(n_points, 2)
print(individual)
points = []

radius = 0.5
p0 = (0.5 + radius * (individual[0][0] - 0.5), 0.5 + radius * (individual[0][1] - 0.5))
points.append(p0)
for i in range(1, n_points):
    radius = 0.1
    p1 = (p0[0] + radius * (individual[i][0] - 0.5), p0[1] + radius * (individual[i][1] - 0.5))
    p0 = p1
    points.append(p1)

points = torch.tensor(points)
points[:, 0] *= img_size
points[:, 1] *= img_size

points[:, 0] /= img_size
points[:, 1] /= img_size
new_points = []
for p, point in enumerate(points):
    if p == 0:
        radius = 0.5
        p_x = ((points[p][0] - 0.5) / radius) + 0.5
        p_y = ((points[p][1] - 0.5) / radius) + 0.5
        new_points.append([p_x, p_y])
    else:
        radius = 0.1
        p_x = ((points[p][0] - points[p-1][0]) / radius) + 0.5
        p_y = ((points[p][1] - points[p-1][1]) / radius) + 0.5
        new_points.append([p_x, p_y])

new_points = np.array(new_points)

print(points)
print(new_points)


points = []
radius = 0.5
p0 = (0.5 + radius * (new_points[0][0] - 0.5), 0.5 + radius * (new_points[0][1] - 0.5))
points.append(p0)
for i in range(1, n_points):
    radius = 0.1
    p1 = (p0[0] + radius * (new_points[i][0] - 0.5), p0[1] + radius * (new_points[i][1] - 0.5))
    p0 = p1
    points.append(p1)

print(points)


individual = np.random.randn(n_points, 2)
print("Original", individual)
individual = individual.flatten()
print("After flatten", individual)
individual = individual.reshape((n_points, 2))
print("Reconstruction", individual)
