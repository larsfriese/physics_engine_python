import numpy as np
import math
from sympy import symbols, diff
from sympy.parsing.sympy_parser import parse_expr
from objects import Particle


class ConstraintManager:
    def __init__(self, scene: list, dimensions: int):
        self.scene = scene
        self.dimensions = dimensions

        # list of positions, velocities and forces
        scene_length = len(self.scene)
        self.q = np.zeros(self.dimensions * scene_length, dtype=np.float64).reshape((self.dimensions * scene_length, 1))
        self.dq = np.zeros(self.dimensions * scene_length, dtype=np.float64).reshape((self.dimensions * scene_length, 1))
        self.Q = np.zeros(self.dimensions * scene_length, dtype=np.float64).reshape((self.dimensions * scene_length, 1))

        # create the diagonal mass matrix
        dim_times_particles = self.dimensions * scene_length
        self.M = np.zeros(shape=(dim_times_particles, dim_times_particles), dtype=np.float64)

        for c, particle in enumerate(scene):
            for dim in range(self.dimensions):
                self.q[2*c+dim] = particle.position[dim]
                self.dq[2*c+dim] = particle.velocity[dim]
                self.Q[2*c+dim] = particle.force_accumulator[dim]
                self.M[2*c+dim, 2*c+dim] = particle.mass

        self.W = np.linalg.inv(self.M)                                         # inverse mass matrix

        self.j = np.zeros(shape=(0, dim_times_particles), dtype=np.float64)      # jacobi matrix J = δC/δq
        self.dj = np.zeros(shape=(0, dim_times_particles), dtype=np.float64)     # derivative of the jacobi matrix

        self.c = []                                                              # constraint C
        self.dc = []                                                             # derivative of the constraints δC/dδ

    def update(self) -> None:
        self.__init__(self.scene, self.dimensions)

    def rail_constraint(self, particle: Particle, function: str) -> None:

        # find index of object in scene
        index = self.scene.index(particle)
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
                j.extend([-df.subs(x, particle.position[0]), 1])
            else:
                j.extend([0, 0])

        j = np.array(j, dtype=np.float64).reshape((1, objects*2))

        # reshape the arrays so that it's a row vector instead of a column vector
        # also the data type is changed to float64
        dj = []
        for c, i in enumerate(range(objects)):
            if c == index:
                dj.extend([-ddf.subs(x, particle.position[0]) * particle.velocity[0], 0])
            else:
                dj.extend([0, 0])

        dj = np.array(dj, dtype=np.float64).reshape((1, objects*2))

        self.j = np.concatenate((self.j, j), axis=0)
        self.dj = np.concatenate((self.dj, dj), axis=0)

        self.c += [particle.position[1] - f.subs(x, particle.position[0])]
        self.dc += [-df.subs(x, particle.position[0]) * particle.velocity[0] + particle.velocity[1]]

    def distance_constraint(self, particle1: Particle, particle2: Particle) -> None:

        # find index of object in scene
        index1 = self.scene.index(particle1)
        index2 = self.scene.index(particle2)
        objects = len(self.scene)

        # positions
        x1 = particle1.position[0]
        y1 = particle1.position[1]
        x2 = particle2.position[0]
        y2 = particle2.position[1]

        # velocities
        x1_vel = particle1.velocity[0]
        y1_vel = particle1.velocity[1]
        x2_vel = particle2.velocity[0]
        y2_vel = particle2.velocity[1]

        # denominator for all the derivatives
        u = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

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

    def circular_wire_constraint(self, particle: Particle) -> None:

        # find index of object in scene
        index = self.scene.index(particle)
        objects = len(self.scene)

        # positions
        x = particle.position[0]
        y = particle.position[1]

        # velocities
        x_vel = particle.velocity[0]
        y_vel = particle.velocity[1]

        # denominator for all the derivatives
        u = np.sqrt(x**2 + y**2)

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

    def add_forces(self):
        j_trans = self.j.T
        """p1 = self.j.dot(self.W).dot(j_trans)
        p2 = self.dj.dot(self.dq)
        p3 = self.j.dot(self.W).dot(self.Q)"""
        solution_vectors = np.linalg.solve(self.j.dot(self.W).dot(j_trans),
                                           - self.dj.dot(self.dq) - self.j.dot(self.W).dot(self.Q))
        constraint = j_trans.dot(solution_vectors)

        # add forces
        for c, particle in enumerate(self.scene):
            """i.force_accumulator_x += constraint[2*c][0]
            i.force_accumulator_y += constraint[2*c+1][0]"""
            particle.force_accumulator[0] += constraint[2*c][0]
            particle.force_accumulator[1] += constraint[2*c+1][0]

    def get_forces(self):
        j_trans = self.j.T
        solution_vectors = np.linalg.solve(self.j.dot(self.W).dot(j_trans),
                                           - self.dj.dot(self.dq) - self.j.dot(self.W).dot(self.Q))
        return j_trans.dot(solution_vectors)
