import numpy as np
import pylab as plt


def dist(x1, x2, p):
    """
    Returns distance and closest point on line between x1 and x2 from point p
    """
    px = x2[0]-x1[0]
    py = x2[1]-x1[1]
    pz = x2[2]-x1[2]

    delta = px*px + py*py + pz*pz

    u = ((p[0] - x1[0]) * px + (p[1] - x1[1]) * py + (p[2] - x1[2]) * pz) / float(delta)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    x = x1[0] + u * px
    y = x1[1] + u * py
    z = x1[2] + u * pz

    closest_point = [x, y, z]
    dx = x - p[0]
    dy = y - p[1]
    dz = z - p[2]
    dist = np.sqrt(dx*dx + dy*dy + dz*dz)

    return dist, closest_point

r_limit = 5
x1 = np.array([-0, 0, 0])
x2 = np.array([0, 0, 10])
x = np.array([40, 0, 5])

nx = 51
ny = 51
nz = 51

xs, zs = np.meshgrid(np.linspace(-50, 50, nx), np.linspace(-50, 50, nz))

xs = xs.flatten()
zs = zs.flatten()
# ys = np.zeros(len(xs))

move_xz = np.zeros(xs.shape)
move_xy = np.zeros(xs.shape)

for i in range(xs.shape[0]):
    distance, closest_point = dist(x1,x2, [xs[i], 0, zs[i]])
    if distance < r_limit:
        move_xz[i] = 1

for i in range(xs.shape[0]):
    distance, closest_point = dist(x1,x2, [xs[i], zs[i], 0])
    if distance < r_limit:
        move_xy[i] = 1

distance, closest_point = dist(x1, x2, x)


plt.subplot(121, aspect=1)
plt.pcolormesh(xs.reshape(nz, nx), zs.reshape(nz, nx), move_xy.reshape(nz, nx))
plt.plot([x1[0], x2[0]], [x1[1], x2[1]], 'kx-')
plt.plot([x[0], closest_point[0]], [x[1], closest_point[1]], 'ro-')
plt.subplot(122, aspect=1)
plt.pcolormesh(xs.reshape(nz, nx), zs.reshape(nz, nx), move_xz.reshape(nz, nx))
plt.plot([x1[0], x2[0]], [x1[2], x2[2]], 'kx-')
plt.plot([x[0], closest_point[0]], [x[2], closest_point[2]], 'ro-')

plt.plot(x[0], x[2], 'o')



plt.savefig('linetest.png')