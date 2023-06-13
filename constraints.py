import numpy as np
import time, math
from sympy import symbols, diff
from sympy.parsing.sympy_parser import parse_expr


def evaluate(function):
    start = time.time()
    function()
    end = time.time()
    return end - start


class ConstraintManager:
    def __init__(self, scene):
        self.scene = scene
        self.dim = 2

        # list of positions, velocities and forces
        self.q = np.zeros(self.dim * len(scene)).reshape((self.dim * len(scene), 1))
        self.dq = np.zeros(self.dim * len(scene)).reshape((self.dim * len(scene), 1))
        self.Q = np.zeros(self.dim * len(scene)).reshape((self.dim * len(scene), 1))
        for c, i in enumerate(scene):
            self.q[2*c] = i.x
            self.q[2*c+1] = i.y
            self.dq[2*c] = i.x_vel
            self.dq[2*c+1] = i.y_vel
            self.Q[2*c] = i.force_accumulator_x
            self.Q[2*c+1] = i.force_accumulator_y

        # create the diagonal mass matrix
        elements = self.dim * len(self.scene)
        self.M = np.zeros(shape=(elements, elements))
        for c, i in enumerate(scene):
            self.M[2*c, 2*c] = i.mass
            self.M[2*c+1, 2*c+1] = i.mass

        # inverse of mass matrix
        self.W = np.linalg.inv(self.M)

        # create the jacobi matrix
        self.j = np.zeros(shape=(0, elements))
        self.dj = np.zeros(shape=(0, elements))

        # create the constraint vector
        self.c = []
        self.dc = []

    def update(self, scene):
        self.__init__(self.scene)

    def rail_constraint(self, obj, function):

        # find index of object in scene
        index = self.scene.index(obj)
        objects = len(self.scene)

        # parse the function
        x = symbols('x')
        f = parse_expr(function)
        df = diff(f, x)
        ddf = diff(df, x)

        # reshape the arrays so that it's a row vector instead of a column vector
        # also the data type is changed to float64
        j = []
        for c, i in enumerate(range(objects)):
            if c == index:
                j.extend([-df.subs(x, obj.x), 1])
            else:
                j.extend([0, 0])

        j = np.array(j, dtype=np.float64).reshape((1, objects*2))

        # reshape the arrays so that it's a row vector instead of a column vector
        # also the data type is changed to float64
        dj = []
        for c, i in enumerate(range(objects)):
            if c == index:
                dj.extend([-ddf.subs(x, obj.x) * obj.x_vel, 0])
            else:
                dj.extend([0, 0])

        dj = np.array(dj, dtype=np.float64).reshape((1, objects*2))

        self.j = np.concatenate((self.j, j), axis=0)
        self.dj = np.concatenate((self.dj, dj), axis=0)

        self.c += [obj.y - f.subs(x, obj.x)]
        self.dc += [-df.subs(x, obj.x) * obj.x_vel + obj.y_vel]

        return [j, dj]

    def distance_constraint(self, obj1, obj2):

        # find index of object in scene
        index1 = self.scene.index(obj1)
        index2 = self.scene.index(obj2)
        objects = len(self.scene)

        # positions
        x1 = obj1.x
        y1 = obj1.y
        x2 = obj2.x
        y2 = obj2.y

        # velocities
        x1_vel = obj1.x_vel
        y1_vel = obj1.y_vel
        x2_vel = obj2.x_vel
        y2_vel = obj2.y_vel

        # denominator for all the derivatives
        u = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

        j = []
        for c, i in enumerate(range(objects)):
            if c == index1:
                j.extend([(x1 - x2) / u, (y1 - y2) / u])
            elif c == index2:
                j.extend([(-1 * (x1 - x2)) / u, (-1 * (y1 - y2)) / u])
            else:
                j.extend([0, 0])

        # jacobian matrix
        j = np.array(j, dtype=np.float64).reshape((1, objects*2))

        dj = []
        for c, i in enumerate(range(objects)):
            if c == index1:
                dj.extend([(-1 * (x1 * (y1_vel - y2_vel) - x1_vel * (y1 - y2) - x2 * (y1_vel - y2_vel) + x2_vel * (y1 - y2)) * (y1 - y2)) / u**3,
                           ((x1 ** 2) * (y1_vel - y2_vel) - x1 * (x1_vel * (y1 - y2) + 2 * x2 * (y1_vel - y2_vel) - x2_vel * (y1 - y2)) + ((x1_vel * (y1 - y2) + x2 * (y1_vel - y2_vel) - x2_vel * (y1 - y2)) * x2)) / u**3])
            elif c == index2:
                dj.extend([(x1 * (y1_vel - y2_vel) - x1_vel * (y1 - y2) - x2 * (y1_vel - y2_vel) + x2_vel * (y1 - y2)) * (y1 - y2) / u**3,
                           (-1 * ((x1 ** 2) * (y1_vel - y2_vel) - x1 * (x1_vel * (y1 - y2) + 2 * x2 * (y1_vel - y2_vel) - x2_vel * (y1 - y2)) + ((x1_vel * (y1 - y2) + x2 * (y1_vel - y2_vel) - x2_vel * (y1 - y2)) * x2))) / u**3])
            else:
                dj.extend([0, 0])

        # derivative of the jacobi matrix
        dj = np.array(dj, dtype=np.float64).reshape((1, objects*2))

        self.j = np.concatenate((self.j, j), axis=0)
        self.dj = np.concatenate((self.dj, dj), axis=0)

        return [j, dj]

    def circular_wire_constraint(self, obj):

        # find index of object in scene
        index = self.scene.index(obj)
        objects = len(self.scene)

        # positions
        x = obj.x
        y = obj.y

        # velocities
        x_vel = obj.x_vel
        y_vel = obj.y_vel

        # denominator for all the derivatives
        u = math.sqrt(x**2 + y**2)

        j = []
        for c, i in enumerate(range(objects)):
            if c == index:
                j.extend([x / u, y / u])
            else:
                j.extend([0, 0])

        # jacobi matrix
        j = np.array(j, dtype=np.float64).reshape((1, objects*2))

        dj = []
        for c, i in enumerate(range(objects)):
            if c == index:
                dj.extend([(x_vel * (y ** 2) - y_vel * x * y) / u**3,
                           (y_vel * (x ** 2) - x_vel * x * y) / u**3])
            else:
                dj.extend([0, 0])

        # derivative of the jacobi matrix
        dj = np.array(dj, dtype=np.float64).reshape((1, objects*2))

        self.j = np.concatenate((self.j, j), axis=0)
        self.dj = np.concatenate((self.dj, dj), axis=0)

        return [j, dj]

    def add_forces(self):
        j_trans = self.j.T
        p1 = self.j.dot(self.W).dot(j_trans)
        p2 = self.dj.dot(self.dq)
        p3 = self.j.dot(self.W).dot(self.Q)
        solution_vectors = np.linalg.solve(self.j.dot(self.W).dot(j_trans), - self.dj.dot(self.dq) - self.j.dot(self.W).dot(self.Q))
        constraint = j_trans.dot(solution_vectors)

        # add forces
        for c, i in enumerate(self.scene):
            i.force_accumulator_x += constraint[2*c][0]
            i.force_accumulator_y += constraint[2*c+1][0]

    def get_forces(self):
        j_trans = self.j.T
        solution_vectors = np.linalg.solve(self.j.dot(self.W).dot(j_trans),
                                           - self.dj.dot(self.dq) - self.j.dot(self.W).dot(self.Q))
        return j_trans.dot(solution_vectors)