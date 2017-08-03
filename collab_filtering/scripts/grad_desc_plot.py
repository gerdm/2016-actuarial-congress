import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D


fxy = lambda x, y: 0.4*x**2 + 0.3*y**2+0.1*x*y
pfx = lambda x, y: 0.8*x + 0.1*y
pfy = lambda x, y: 0.6*y + 0.1*x

Fxy = np.vectorize(fxy)
pFx = np.vectorize(pfx)
pFy = np.vectorize(pfy)

class Roll(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    
    def update_pos(self, nx, ny, nz):
        self.x.append(nx)
        self.y.append(ny)
        self.z.append(nz)

    def pos_out(self):
        return (self.x[-1], self.y[-1], self.z[-1])

    def remove_points(self):
        self.x = []
        self.y = []
        self.z = []


x0 = 4
y0 = 5
z0 = fxy(x0, y0)

x03, y03 = 5, -2
z03 = fxy(x03, y03)

x04, y04 = 4, -4
z04 = fxy(x04, y04)

roll1 = Roll(x=[x0], y=[y0], z=[z0])
roll2 = Roll(x=[x0], y=[y0], z=[z0])
roll3 = Roll(x=[x03], y=[y03], z=[z03])
roll4 = Roll(x=[x04], y=[y04], z=[z04])


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x, y = np.linspace(-5, 5, 150), np.linspace(-5, 5, 150)
x, y = np.meshgrid(x, y)
z = Fxy(x, y)

max_frames = 300

def compute_new_pos(dx, dy, dz, alpha):
    out_dx = dx - alpha * pfx(dx, dy)
    out_dy = dy - alpha * pfy(dx, dy)
    out_dz = fxy(dx, dy)
    return out_dx, out_dy, out_dz

def compute_update(roll, alpha, color="r"):
    dx, dy, dz = roll.pos_out()
    step = ax.plot3D([dx], [dy], [dz], "o", alpha=0.6, color=color)
    dx, dy, dz = compute_new_pos(dx, dy, dz, alpha)
    roll.update_pos(dx, dy, dz)
    return step

def gradient_desc(i):
    ax.clear()
    ax.set_title(r"$0.4x^2 + 0.3y^2 + 0.1xy$")
    ax.view_init(45, 180 + 0.4 * i)
    surface = ax.plot_surface(x, y, z, cmap=plt.cm.BrBG, alpha=0.6)

    step1 = compute_update(roll1, 0.009, "r")
    step2 = compute_update(roll2, 0.05, "b")
    step3 = compute_update(roll3, 0.01, "g")
    step4 = compute_update(roll4, 0.015, "m")

    if i == (max_frames - 1):
        roll1.x, roll1.y, roll1.z = [x0], [y0], [z0]
        roll2.x, roll2.y, roll2.z = [x0], [y0], [fxy(x0, y0)]
        roll3.x, roll3.y, roll3.z = [x03], [y03], [z03]
        roll4.x, roll4.y, roll4.z = [x04], [y04], [z04]
    return step1, step2, step3, step4

animate = animation.FuncAnimation(fig, gradient_desc, interval=30, frames = max_frames, blit=False)
#animate.save('gd.mp4', fps=15, extra_args=['-vcodec', 'libx264'])
animate.save('gd.gif', fps=60, writer='imagemagick')
plt.show()
