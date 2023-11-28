import numpy as np


class LinearFrictionForce:
    def __init__(self, scene: list, strength: float, dimensions: int):
        self.strength = np.float64(strength)
        self.scene = scene
        self.dimension = dimensions

    def add_forces(self) -> None:
        for particle in self.scene:
            for dim in range(self.dimension):
                particle.force_accumulator[dim] += -self.strength * particle.velocity[dim]


class Gravity:
    def __init__(self, scene: list, strength: float, dimension: int):
        self.strength = np.float64(strength)
        self.scene = scene
        self.dimension = dimension

    def add_forces(self) -> None:
        for i in self.scene:
            # apply force in the specified dimension, e.g. y dimension
            i.force_accumulator[self.dimension] += -self.strength * i.mass

    def potential_energy(self) -> float:
        potential_energy = 0
        for i in self.scene:
            potential_energy += i.mass * self.strength * i.position[self.dimension]
        return potential_energy
