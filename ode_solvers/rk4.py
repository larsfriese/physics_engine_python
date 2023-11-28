import numpy as np
from objects import Particle


def rk4_1(p1: Particle) -> tuple:
    dt = p1.timestep
    kx0 = p1.velocity * dt
    kv0 = (p1.force_accumulator / p1.mass) * dt
    kx1 = (p1.velocity + kv0 / 2.00) * dt
    p1.force_accumulator = np.array([0, 0], dtype=np.float64)
    return kx0, kv0, kx1


def rk4_2(p1: Particle, org_vel: np.ndarray) -> tuple:
    dt = p1.timestep
    kv1 = (p1.force_accumulator / p1.mass) * dt
    kx2 = (org_vel + kv1 / 2.00) * dt
    p1.force_accumulator = np.array([0, 0], dtype=np.float64)
    return kx2, kv1


def rk4_3(p1: Particle, org_vel: np.ndarray) -> tuple:
    dt = p1.timestep
    kv2 = (p1.force_accumulator / p1.mass) * dt
    kx3 = (org_vel + kv2) * dt
    p1.force_accumulator = np.array([0, 0], dtype=np.float64)
    return kx3, kv2


def rk4_4(p1: Particle) -> np.ndarray:
    dt = p1.timestep
    kv3 = (p1.force_accumulator / p1.mass) * dt
    p1.force_accumulator = np.array([0, 0], dtype=np.float64)
    return kv3


def apply_rk4(p1: Particle, kx0: np.ndarray, kx1: np.ndarray, kx2: np.ndarray, kx3: np.ndarray,
              kv0: np.ndarray, kv1: np.ndarray, kv2: np.ndarray, kv3: np.ndarray,
              org_coord: np.ndarray, org_vel: np.ndarray) -> None:
    p1.position = org_coord + (kx0 + 2.0 * kx1 + 2.0 * kx2 + kx3) / 6.0
    p1.velocity = org_vel + (kv0 + 2.0 * kv1 + 2.0 * kv2 + kv3) / 6.0
    p1.trail.append(p1.position)
    p1.force_accumulator = np.array([0, 0], dtype=np.float64)


def runge_kutta_4th_order(particles, add_forces):
    """
    This function takes a list of objects and a function that adds forces to the objects.
    It then calculates the next step in the simulation using the RK4 method.

    :param particles: list of particles
    :param add_forces: function that adds forces to the objects
    :return: None
    """

    original_coordinates = [particle.position for particle in particles]
    original_velocities = [particle.velocity for particle in particles]

    add_forces()

    # first step
    kx0_list = []
    kx1_list = []
    kv0_list = []

    for index, particle in enumerate(particles):
        kx0, kv0, kx1 = rk4_1(particle)
        kx0_list.append(kx0)
        kx1_list.append(kx1)
        kv0_list.append(kv0)

    for index, particle in enumerate(particles):
        particle.position = original_coordinates[index] + kx0_list[index] / 2
        particle.velocity = original_velocities[index] + kv0_list[index] / 2

    add_forces()

    # second step
    kx2_list = []
    kv1_list = []

    for index, particle in enumerate(particles):
        kx2, kv1 = rk4_2(particle, original_velocities[index])
        kx2_list.append(kx2)
        kv1_list.append(kv1)

    for index, particle in enumerate(particles):
        particle.position = original_coordinates[index] + kx1_list[index] / 2
        particle.velocity = original_velocities[index] + kv1_list[index] / 2

    add_forces()

    # third step
    kx3_list = []
    kv2_list = []

    for index, particle in enumerate(particles):
        kx3, kv2 = rk4_3(particle, original_velocities[index])
        kx3_list.append(kx3)
        kv2_list.append(kv2)

    for index, particle in enumerate(particles):
        particle.position = original_coordinates[index] + kx2_list[index]
        particle.velocity = original_velocities[index] + kv2_list[index]

    add_forces()

    # fourth step
    kv3_list = []

    for particle in particles:
        kv3 = rk4_4(particle)
        kv3_list.append(kv3)

    for index, particle in enumerate(particles):
        apply_rk4(particle, kx0_list[index], kx1_list[index], kx2_list[index], kx3_list[index],
                  kv0_list[index], kv1_list[index], kv2_list[index], kv3_list[index], original_coordinates[index],
                  original_velocities[index])
