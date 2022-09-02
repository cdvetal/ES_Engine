from utils import Vector, perpendicular, normalize


A = Vector(0.0, 0.0)
B = Vector(0.0, 0.0)

AB = B - A

AB_perp_normed = perpendicular(normalize(AB))

print(AB_perp_normed)
