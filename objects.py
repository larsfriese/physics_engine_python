import pygame
import math
import numpy as np
from numpy import float64

WHITE = (255, 255, 255)


def coords_to_pygame(coords: tuple) -> tuple:
    return coords[0] + 400, -coords[1] + 400


class Particle:
    def __init__(self, position: list, mass: float, dimensions: int, zoom: int,
                 timestep: float, radius: float, color: tuple):

        self.position = np.array(position, dtype=float64)
        self.velocity = np.array(dimensions * [0], dtype=float64)           # initializing all velocities to 0
        self.previous_position = np.array(position, dtype=float64)
        self.mass = np.float64(mass)
        self.force_accumulator = np.array(dimensions * [0], dtype=float64)  # initializing all forces to 0
        self.timestep = np.float64(timestep)
        self.dimensions = dimensions

        self.radius = radius
        self.color = color
        self.trail = []
        self.ZOOM = zoom

        """self.x = x
        self.y = y
        self.x_vel = 0
        self.y_vel = 0

        self.prev_x = x
        self.prev_y = y

        self.mass = mass
        self.force_accumulator_x = 0
        self.force_accumulator_y = 0
        self.timestep = timestep

        self.radius = radius
        self.color = color
        self.trail = []"""

    def distance(self, tuple_coords: tuple) -> float:
        temporary_sum = 0
        for i in range(self.dimensions):
            temporary_sum += (self.position[i] - tuple_coords[i]) ** 2
        return np.sqrt(temporary_sum)

    def draw(self, win, trail=True):

        if trail:
            # trail of object
            if len(self.trail) > 2:
                updated_points = []
                for point in self.trail:
                    x, y = point[0], point[1]
                    updated_points.append(coords_to_pygame((self.ZOOM * x, self.ZOOM * y)))

                pygame.draw.lines(win, self.color, False, updated_points, 2)

        pygame.draw.circle(win, self.color, coords_to_pygame((self.ZOOM * self.position[0], self.ZOOM * self.position[1])), self.ZOOM * self.radius)

    def euler_method(self):
        current_velocity = self.velocity
        self.position += current_velocity * self.timestep
        self.velocity += self.force_accumulator / self.mass * self.timestep
        self.trail.append((self.x, self.y))
        self.force_accumulator = np.array([0, 0])
        """x_vel = self.x_vel
        y_vel = self.y_vel
        self.x += x_vel * self.timestep
        self.y += y_vel * self.timestep
        self.x_vel += self.force_accumulator_x / self.mass * self.timestep
        self.y_vel += self.force_accumulator_y / self.mass * self.timestep
        self.trail.append((self.x, self.y))
        self.force_accumulator_x = 0
        self.force_accumulator_y = 0"""

    def semi_implicit_euler(self):
        self.position += self.velocity * self.timestep
        self.velocity += (self.force_accumulator / self.mass) * self.timestep
        self.trail.append((self.position[0], self.position[1]))
        self.force_accumulator = np.array([0, 0])
        """self.x += self.x_vel * self.timestep
        self.y += self.y_vel * self.timestep
        self.x_vel += self.force_accumulator_x / self.mass * self.timestep
        self.y_vel += self.force_accumulator_y / self.mass * self.timestep
        self.trail.append((self.x, self.y))
        self.force_accumulator_x = 0
        self.force_accumulator_y = 0"""


    def verlet(self):
        dt = self.timestep

        temp_x = self.x
        temp_y = self.y

        self.x = 2 * self.x - self.prev_x + (self.force_accumulator_x / self.mass) * (dt ** 2)
        self.y = 2 * self.y - self.prev_y + (self.force_accumulator_y / self.mass) * (dt ** 2)

        self.prev_x = temp_x
        self.prev_y = temp_y

        self.x_vel += (self.force_accumulator_x / self.mass) * dt
        self.y_vel += (self.force_accumulator_y / self.mass) * dt

        self.trail.append((self.x, self.y))
        self.force_accumulator_x = 0
        self.force_accumulator_y = 0

    def velocity_verlet_1(self):
        dt = self.timestep

        self.x += self.x_vel * dt + 0.5 * (self.force_accumulator_x / self.mass) * (dt ** 2)
        self.y += self.y_vel * dt + 0.5 * (self.force_accumulator_y / self.mass) * (dt ** 2)

        v_x = self.x_vel + 0.5 * (self.force_accumulator_x / self.mass) * dt
        v_y = self.y_vel + 0.5 * (self.force_accumulator_y / self.mass) * dt

        self.force_accumulator_x = 0
        self.force_accumulator_y = 0

        return [v_x, v_y]

    def velocity_verlet_2(self, velocities):
        dt = self.timestep

        new_a_x = self.force_accumulator_x / self.mass
        new_a_y = self.force_accumulator_y / self.mass

        self.x_vel = velocities[0] + 0.5 * new_a_x * dt
        self.y_vel = velocities[1] + 0.5 * new_a_y * dt

        self.trail.append((self.x, self.y))

    def energy(self) -> float:
        return 0.5 * self.mass * np.linalg.norm(self.velocity) ** 2


class Spring:
    def __init__(self, p1, p2, length, k):
        self.p1 = p1
        self.p2 = p2
        self.length = length
        self.k = k

    def draw(self, win):
        pygame.draw.line(win, WHITE, coords_to_pygame((self.p1.position[0], self.p1.position[1])), coords_to_pygame((self.p2.position[0], self.p2.position[1])), 2)

    def add_forces(self):
        distance_x = self.p1.position[0] - self.p2.position[0]
        distance_y = self.p1.position[1] - self.p2.position[1]
        distance = math.sqrt(distance_x ** 2 + distance_y ** 2)

        force = (self.k * (self.length - distance))
        theta = math.atan2(distance_y, distance_x)
        force_x = math.cos(theta) * force
        force_y = math.sin(theta) * force

        self.p1.force_accumulator[0] += force_x
        self.p1.force_accumulator[1] += force_y

        self.p2.force_accumulator[0] += -force_x
        self.p2.force_accumulator[1] += -force_y

    def calc_force(self):
        distance_x = self.p1.position[0] - self.p2.position[0]
        distance_y = self.p1.position[1] - self.p2.position[1]
        distance = math.sqrt(distance_x ** 2 + distance_y ** 2)

        force = (self.k * (self.length - distance))
        theta = math.atan2(distance_y, distance_x)
        force_x = math.cos(theta) * force
        force_y = math.sin(theta) * force

        return [force_x, force_y]

    def energy(self):
        # calculate the distance between the two objects
        distance_x = self.p1.position[0] - self.p2.position[0]
        distance_y = self.p1.position[1] - self.p2.position[1]
        distance = math.sqrt(distance_x ** 2 + distance_y ** 2)
        return 0.5 * self.k * (distance - self.length) ** 2


class Spring_to_mouse:
    def __init__(self, p1, length, k, mouse_x, mouse_y):
        self.p1 = p1
        self.length = length
        self.k = k
        self.mouse_x = mouse_x
        self.mouse_y = mouse_y

    def draw(self, win):
        pygame.draw.line(win, WHITE, coords_to_pygame([self.p1.x, self.p1.y]), [self.mouse_x, self.mouse_y], 2)

    def add_forces(self):
        p1x, p1y = coords_to_pygame([self.p1.x, self.p1.y])
        distance_x = p1x - self.mouse_x
        distance_y = p1y - self.mouse_y
        distance = math.sqrt(distance_x ** 2 + distance_y ** 2)

        force = (self.k * (self.length - distance))
        theta = math.atan2(distance_y, distance_x)
        force_x = math.cos(theta) * force
        force_y = math.sin(theta) * force

        self.p1.force_accumulator_x += force_x
        self.p1.force_accumulator_y += force_y

    def calc_force(self):
        p1x, p1y = coords_to_pygame([self.p1.x, self.p1.y])
        distance_x = p1x - self.mouse_x
        distance_y = p1y - self.mouse_y
        distance = math.sqrt(distance_x ** 2 + distance_y ** 2)

        force = (self.k * (self.length - distance))
        theta = math.atan2(distance_y, distance_x)
        force_x = math.cos(theta) * force
        force_y = math.sin(theta) * force

        return [force_x, force_y]

    def energy(self):
        p1x, p1y = coords_to_pygame([self.p1.x, self.p1.y])

        # calculate the distance between the two objects
        distance_x = p1x - self.mouse_x
        distance_y = p1y - self.mouse_y
        distance = math.sqrt(distance_x ** 2 + distance_y ** 2)
        return 0.5 * self.k * (distance - self.length) ** 2