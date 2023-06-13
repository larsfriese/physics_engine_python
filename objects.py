import pygame
import math

WHITE = (255, 255, 255)


def coords_to_pygame(coords):
    return coords[0] + 400, -coords[1] + 400


class Object:
    def __init__(self, x, y, radius, color, mass, timestep):
        self.x = x
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
        self.trail = []

    def distance(self, touple_coords):
        return math.sqrt((self.x - touple_coords[0]) ** 2 + (self.y - touple_coords[1]) ** 2)

    def draw(self, win, trail=True):

        if trail:
            # trail of object
            if len(self.trail) > 2:
                updated_points = []
                for point in self.trail:
                    x, y = point
                    updated_points.append(coords_to_pygame((x, y)))

                pygame.draw.lines(win, self.color, False, updated_points, 2)

        pygame.draw.circle(win, self.color, coords_to_pygame([self.x, self.y]), self.radius)

    def euler_method(self):
        x_vel = self.x_vel
        y_vel = self.y_vel
        self.x += x_vel * self.timestep
        self.y += y_vel * self.timestep
        self.x_vel += self.force_accumulator_x / self.mass * self.timestep
        self.y_vel += self.force_accumulator_y / self.mass * self.timestep
        self.trail.append((self.x, self.y))
        self.force_accumulator_x = 0
        self.force_accumulator_y = 0

    def semi_implicit_euler(self):
        self.x += self.x_vel * self.timestep
        self.y += self.y_vel * self.timestep
        self.x_vel += self.force_accumulator_x / self.mass * self.timestep
        self.y_vel += self.force_accumulator_y / self.mass * self.timestep
        self.trail.append((self.x, self.y))
        self.force_accumulator_x = 0
        self.force_accumulator_y = 0

    def rk(self):
        self.x_vel += 0.5 * (self.force_accumulator_x / self.mass) * self.timestep
        self.y_vel += 0.5 * (self.force_accumulator_y / self.mass) * self.timestep

        self.x += self.x_vel * self.timestep
        self.y += self.y_vel * self.timestep

        self.trail.append((self.x, self.y))
        self.force_accumulator_x = 0
        self.force_accumulator_y = 0

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

    def rk4_1(self):
        dt = self.timestep

        kx0x = self.x_vel * dt
        kx0y = self.y_vel * dt

        kv0x = (self.force_accumulator_x / self.mass) * dt
        kv0y = (self.force_accumulator_y / self.mass) * dt

        kx1x = (self.x_vel + 0.5 * kv0x) * dt
        kx1y = (self.y_vel + 0.5 * kv0y) * dt

        self.force_accumulator_x = 0
        self.force_accumulator_y = 0

        return kx0x, kv0x, kx1x, kx0y, kv0y, kx1y

    def rk4_2(self, kv1x, kv1y, vt):
        dt = self.timestep

        kv1x = (self.force_accumulator_x / self.mass) * dt
        kv1y = (self.force_accumulator_y / self.mass) * dt

        kx2x = (vt[0] + 0.5 * kv1x) * dt
        kx2y = (vt[1] + 0.5 * kv1y) * dt

        self.force_accumulator_x = 0
        self.force_accumulator_y = 0

        return [kx2x, kv1x, kx2y, kv1y]

    def rk4_3(self, vt):
        dt = self.timestep

        kv2x = (self.force_accumulator_x / self.mass) * dt
        kv2y = (self.force_accumulator_y / self.mass) * dt

        kx3x = (vt[0] + kv2x) * dt
        kx3y = (vt[1] + kv2y) * dt

        self.force_accumulator_x = 0
        self.force_accumulator_y = 0

        return [kx3x, kv2x, kx3y, kv2y]

    def rk4_4(self):
        dt = self.timestep
        kv3x = (self.force_accumulator_x / self.mass) * dt
        kv3y = (self.force_accumulator_y / self.mass) * dt

        self.force_accumulator_x = 0
        self.force_accumulator_y = 0

        return [kv3x, kv3y]

    def rk4_final(self, args):
        original_coords, original_vels, kx0, kx1, kx2, kx3, kv0, kv1, kv2, kv3 = args
        kx0x, kx0y = kx0
        kx1x, kx1y = kx1
        kx2x, kx2y = kx2
        kx3x, kx3y = kx3

        kv0x, kv0y = kv0
        kv1x, kv1y = kv1
        kv2x, kv2y = kv2
        kv3x, kv3y = kv3

        self.x = original_coords[0] + (1/6) * (kx0x + 2 * kx1x + 2 * kx2x + kx3x)
        self.y = original_coords[1] + (1/6) * (kx0y + 2 * kx1y + 2 * kx2y + kx3y)

        self.x_vel = original_vels[0] + (1/6) * (kv0x + 2 * kv1x + 2 * kv2x + kv3x)
        self.y_vel = original_vels[1] + (1/6) * (kv0y + 2 * kv1y + 2 * kv2y + kv3y)

        self.trail.append((self.x, self.y))
        self.force_accumulator_x = 0
        self.force_accumulator_y = 0

    def energy(self):
        return 0.5 * self.mass * (self.x_vel ** 2 + self.y_vel ** 2)


class Spring:
    def __init__(self, p1, p2, length, k):
        self.p1 = p1
        self.p2 = p2
        self.length = length
        self.k = k

    def draw(self, win):
        pygame.draw.line(win, WHITE, coords_to_pygame([self.p1.x, self.p1.y]), coords_to_pygame([self.p2.x, self.p2.y]), 2)

    def add_forces(self):
        distance_x = self.p1.x - self.p2.x
        distance_y = self.p1.y - self.p2.y
        distance = math.sqrt(distance_x ** 2 + distance_y ** 2)

        force = (self.k * (self.length - distance))
        theta = math.atan2(distance_y, distance_x)
        force_x = math.cos(theta) * force
        force_y = math.sin(theta) * force

        self.p1.force_accumulator_x += force_x
        self.p1.force_accumulator_y += force_y

        self.p2.force_accumulator_x += -force_x
        self.p2.force_accumulator_y += -force_y

    def calc_force(self):
        distance_x = self.p1.x - self.p2.x
        distance_y = self.p1.y - self.p2.y
        distance = math.sqrt(distance_x ** 2 + distance_y ** 2)

        force = (self.k * (self.length - distance))
        theta = math.atan2(distance_y, distance_x)
        force_x = math.cos(theta) * force
        force_y = math.sin(theta) * force

        return [force_x, force_y]

    def energy(self):
        # calculate the distance between the two objects
        distance_x = self.p1.x - self.p2.x
        distance_y = self.p1.y - self.p2.y
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

class Gravity:
    def __init__(self, scene, strength):
        self.strength = strength
        self.scene = scene

    def add_forces(self):
        for i in self.scene:
            i.force_accumulator_y += - self.strength * i.mass

    def potential_energy(self):
        potential_energy = 0
        for i in self.scene:
            potential_energy += i.mass * self.strength * i.y

        return potential_energy
