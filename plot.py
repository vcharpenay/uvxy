from functools import reduce
from random import random
from math import isnan, isinf, nan, inf
import matplotlib.pyplot as plt

# TODO min/max already have vararg signature
def _max(*args): return reduce(max, args)
def _min(*args): return reduce(min, args)

class Octa:

    def __init__(
            self,
            xmin,
            xmax,
            ymin,
            ymax,
            umin,
            umax,
            vmin,
            vmax
    ) -> None:
        if xmin > xmax or ymin > ymax or umin > umax or vmin > vmax:
            self.is_empty = True
            self.constraints = (nan, nan, nan, nan, nan, nan, nan, nan)

        else:
            self.is_empty = False
            # tighten bounds
            self.constraints = (
                _max(xmin, vmin - ymax, ymin - umax, (vmin - umax) / 2),
                _min(xmax, vmax - ymin, ymax - umin, (vmax - umin) / 2),
                _max(ymin, umin + xmin, vmin - xmax, (umin + vmin) / 2),
                _min(ymax, umax + xmax, vmax - xmin, (umax + vmax) / 2),
                _max(umin, ymin - xmax, vmin - 2 * xmax, 2 * ymin - vmax),
                _min(umax, ymax - xmin, vmax - 2 * xmin, 2 * ymax - vmin),
                _max(vmin, xmin + ymin, umin + 2 * xmin, 2 * ymin - umax),
                _min(vmax, xmax + ymax, umax + 2 * xmax, 2 * ymax - umin)
            )

        xmin, xmax, ymin, ymax, umin, umax, vmin, vmax = self.constraints

    def vertices(self, *, inf_is=10):
        xmin, xmax, ymin, ymax, umin, umax, vmin, vmax = self.constraints

        if isinf(xmin): xmin = -inf_is
        if isinf(xmax): xmax = inf_is
        if isinf(ymin): ymin = -inf_is
        if isinf(ymax): ymax = inf_is
        if isinf(umin): umin = -inf_is
        if isinf(umax): umax = inf_is
        if isinf(vmin): vmin = -inf_is
        if isinf(vmax): vmax = inf_is

        return (
            (xmin, vmin - xmin),
            (xmin, umax + xmin),
            (ymax - umax, ymax),
            (vmax - ymax, ymax),
            (xmax, vmax - xmax),
            (xmax, umin + xmax),
            (ymin - umin, ymin),
            (vmin - ymin, ymin)
        )

    def intersect(self, other):
        if self.is_empty: return self
        elif other.is_empty: return other

        xmin1, xmax1, ymin1, ymax1, umin1, umax1, vmin1, vmax1 = self.constraints
        xmin2, xmax2, ymin2, ymax2, umin2, umax2, vmin2, vmax2 = other.constraints

        return Octa(
            _max(xmin1, xmin2),
            _min(xmax1, xmax2),
            _max(ymin1, ymin2),
            _min(ymax1, ymax2),
            _max(umin1, umin2),
            _min(umax1, umax2),
            _max(vmin1, vmin2),
            _min(vmax1, vmax2)
        )

    def compose(self, other):
        if self.is_empty: return self
        elif other.is_empty: return other

        xmin1, xmax1, ymin1, ymax1, umin1, umax1, vmin1, vmax1 = self.constraints
        xmin2, xmax2, ymin2, ymax2, umin2, umax2, vmin2, vmax2 = other.constraints

        return Octa(
            _max(xmin1, xmin2 - umax1, vmin1 - xmax2),
            _min(xmax1, xmax2 - umin1, vmax1 - xmin2),
            _max(ymin2, umin2 + ymin1, vmin2 - ymax1),
            _min(ymax2, umax2 + ymax1, vmax2 - ymin1),
            _max(ymin2 - xmax1, umin2 + umin1, vmin2 - vmax1),
            _min(ymax2 - xmin1, umax2 + umax1, vmax2 - vmin1),
            _max(xmin1 + ymin2, umin2 + vmin1, vmin2 - umax1),
            _min(xmax1 + ymax2, umax2 + vmax1, vmax2 - umin1)
        )
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Octa): return False

        if self.is_empty and other.is_empty: return True
        elif self.is_empty or other.is_empty: return False

        for c1, c2 in zip(self.constraints, other.constraints):
            if abs(c1 - c2) > 1e-4: return False

        return True

    @staticmethod
    def random_octa():
        # center in [-1,1]*[-1,1]
        cx = random() * 2 - 1
        cy = random() * 2 - 1
        cu  = cy - cx
        cv = cy + cx

        return Octa(
            cx - random(),
            cx + random(),
            cy - random(),
            cy + random(),
            cu - random(),
            cu + random(),
            cv - random(),
            cv + random()
        )

class Parallelogram:

    def __init__(self, umin, umax, vmin, vmax, su, sv) -> None:
        self.constraints = (umin, umax, vmin, vmax, su, sv)

    def vertices(self):
        umin, umax, vmin, vmax, su, sv = self.constraints

        # umax -> f(x) = su*x + umax
        # vmax -> f(x) = -su*x + vmax

        # A = umin/vmin -> su*x + umin = -sv*x + vmin 
        #   -> x = (umin + vmin) / (su + sv)
        #   -> y = (umin + vmin)*su / (su + sv) + umin = 
        # B = umax/vmin
        # C = umax/vmax
        # D = umin/vmax

        xa = (umin + vmin) / (su + sv)
        xb = (umax + vmin) / (su + sv)
        xc = (umax + vmax) / (su + sv)
        xd = (umin + vmax) / (su + sv)

        return (
            (xa, su*xa + umin),
            (xb, su*xb + umax),
            (xc, su*xc + umax),
            (xd, su*xd + umin)
        )

# octa
# base_octa = Octa(-0.6, 0.6, -0.2, 0.6, -0.3, 0.9, -0.5, 1)

# hexa
# base_octa = Octa(-0.8, 0.2, -0.4, 0.6, 0.3, 0.6, -2, 2)

# reflexive-symmetric u-diamond
# base_octa = Octa(-1, 1, -1, 1, -0.2, 0.2, -1, 1)

# u-diamond
# base_octa = Octa(-1, 1, -1, 1, 0.3, 0.6, -1, 1)

# diamond
# base_octa = Octa(-1, 1, -1, 1, 0.1, 0.5, 0.1, 0.6)

# square
# base_octa = Octa(0, 0.5, 0.6, 0.9, -1, 1, 0, 2)

# square (idempotent)
# base_octa = Octa(0, 0.5, 0.3, 0.7, -1, 1, 0, 2)

# octa (idempotent)
# base_octa = Octa(-0.6, 0.3, -0.2, 0.8, 0, 1.4, -0.5, 1)
# base_octa = Octa(0, 0.5, 0.3, 0.7, 0, 0.6, 0.4, 1)

# reflexive-symmetric u-band
# base_octa = Octa(-inf, inf, -inf, inf, -0.1, 0.1, -inf, inf)

# u-band
# base_octa = Octa(-inf, inf, -inf, inf, 0.1, 0.2, -inf, inf)

# v-band
# base_octa = Octa(-inf, inf, -inf, inf, -inf, inf, 0.3, 0.5)

# fully expressive (o-, o+)
# base_octa = Octa(-1, 1, -1, 1, -2, 1, -1, 2)
# base_octa = Octa(-1, 1, -1, 1, -2, 1, -2, 2)

# random
base_octa = Octa.random_octa()

def generate_gif():
    for i in range(10):
        if i == 0:
            octa = base_octa
        else:
            next_octa = octa.compose(base_octa)
            if next_octa == octa: break # TODO keep going for 1/2 steps?
            else: octa = next_octa

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        x, y = zip(*octa.vertices())

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

        # draw equiline
        # ax.plot([-1,1], [-1,1], color='#5e5e5e', linewidth=1)

        # draw octagon
        ax.fill(x, y, facecolor='#5f249f42', edgecolor='#5f249f', linewidth=1)

        # draw constraints
        # ax.axvline(octa[0], color='#5e5e5e', linewidth=1)
        # ax.axvline(octa[1], color='#5e5e5e', linewidth=1)
        # ax.axhline(octa[2], color='#5e5e5e', linewidth=1)
        # ax.axhline(octa[3], color='#5e5e5e', linewidth=1)

        fig.savefig(f"figures/octa-{'0' + str(i) if i < 10 else str(i)}.png")

    # to create a GIF (1s per image):
    # convert -delay 100 -loop 0 *.png animation.gif

def generate_intersection():
    octa1 = Octa(-0.6, 0.6, -0.2, 0.6, -0.3, 0.9, -0.5, 1)
    octa2 = Octa(-1, 1, -1, 1, -0.5, 0.3, -1, 0.7)
    octa3 = octa1.intersect(octa2)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    x1, y1 = zip(*octa1.vertices())
    x2, y2 = zip(*octa2.vertices())
    x3, y3 = zip(*octa3.vertices())

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    ax.fill(x1, y1, facecolor='#87c84610', edgecolor='#87c846', linewidth=1)
    ax.fill(x2, y2, facecolor='#00b9e010', edgecolor='#00b9e0', linewidth=1)
    ax.fill(x3, y3, facecolor='#5f249f42', edgecolor='#00b9e0', linewidth=1)

    fig.savefig(f"figures/octa123.png")

def generate_segment():
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    ax.plot([-0.3, 0.7], [-0.3, 0.7], color='#5e5e5e', linewidth=1)

    fig.savefig(f"figures/octa.png")

def generate_parallelograms():
    p1 = Parallelogram(-0.2, 0.5, 0.1, 1.1, 0.5, 2)
    p2 = Parallelogram(-0.5, 0.3, -0.1, 0.8, 1, 1)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    x1, y1 = zip(*p1.vertices())
    x2, y2 = zip(*p2.vertices())

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    ax.fill(x1, y1, facecolor='#ea760042', edgecolor='#ea7600', linewidth=1)
    ax.fill(x2, y2, facecolor='#bf123842', edgecolor='#bf1238', linewidth=1)

    fig.savefig(f"figures/parallelogram.png")

# generate_gif()
# generate_intersection()
# generate_segment()
generate_parallelograms()
